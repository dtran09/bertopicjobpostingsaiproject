import pandas as pd
import asyncio
from playwright.async_api import async_playwright
from tqdm import tqdm
import random
import time

# === CONFIG ===
INPUT_FILE = "jobs_datascience.csv"
OUTPUT_FILE = "jobs_datascience_full.csv"
CONCURRENT_TABS = 3      # Adjust to 3–8 depending on system
WAIT_SELECTORS = [
    'div[data-test="jobDescriptionText"]',  # Main selector
    'div.desc',                             # Older layout
    'div.jobDescriptionContent',            # Alternate layout
]
MIN_DELAY = 1.0
MAX_DELAY = 3.0
RETRY_LIMIT = 2
SAVE_INTERVAL = 10  # Save every 10 completed jobs

async def fetch_description(page, url):
    """Tries multiple times and selectors to get a job description."""
    for attempt in range(RETRY_LIMIT):
        try:
            # try stronger navigation waits and longer timeout
            await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                # networkidle might time out on some sites; continue anyway
                pass

            # try clicking common "show more" buttons that collapse long descriptions
            for btn_sel in [
                'button[data-test="show-more"]',
                'button[aria-label*="show"]',
                'button[class*="showMore"]',
                'button[class*="see-more"]'
            ]:
                try:
                    btn = await page.query_selector(btn_sel)
                    if btn:
                        await btn.click()
                        await asyncio.sleep(0.5)
                except Exception:
                    continue

            # Try multiple selectors
            for selector in WAIT_SELECTORS:
                try:
                    await page.wait_for_selector(selector, timeout=15000)
                    elem = await page.query_selector(selector)
                    if elem:
                        text = await elem.inner_text()
                        if text and len(text.strip()) > 50:
                            return text.strip()
                except Exception:
                    continue

            # Fallback: try grabbing the main body text and heuristics
            try:
                body_text = await page.inner_text("body")
                if body_text and len(body_text.strip()) > 50:
                    # crude heuristic: return longest contiguous chunk
                    chunks = [c.strip() for c in body_text.split("\n\n") if len(c.strip()) > 50]
                    if chunks:
                        chunks.sort(key=len, reverse=True)
                        return chunks[0]
                    return body_text.strip()
            except Exception:
                pass

            # If still nothing, try scrolling and retrying once
            await page.mouse.wheel(0, 3000)
            await asyncio.sleep(2)
        except Exception:
            # swallow and retry
            await asyncio.sleep(1)
            continue
    return "[Error: Description not found or timeout]"

async def worker(name, queue, results, page, pbar):
    """Each worker owns its Page and processes jobs from the queue."""
    try:
        while True:
            job = await queue.get()
            if job is None:
                queue.task_done()
                break
            idx, url = job
            try:
                desc = await fetch_description(page, url)
                results[idx] = desc
                pbar.update(1)
            except Exception as e:
                results[idx] = f"[Error: {e}]"
                pbar.update(1)
            finally:
                # polite delay per worker to avoid rate-limit
                await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                queue.task_done()
    finally:
        # ensure page is closed by caller/context manager; nothing to do here
        return

async def main():
    df = pd.read_csv(INPUT_FILE)
    if "job_url" not in df.columns:
        print("❌ No job_url column found — cannot fetch descriptions.")
        return

    # Resume logic
    if "description_full" not in df.columns:
        df["description_full"] = df.get("description", "")

    jobs_to_fetch = df[df["description_full"].isna() | (df["description_full"].str.len() < 250)]
    print(f"🧭 Fetching descriptions for {len(jobs_to_fetch)} jobs...")

    if len(jobs_to_fetch) == 0:
        print("✅ All descriptions already fetched.")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ))
        # create one page per worker (no concurrent sharing)
        pages = [await context.new_page() for _ in range(CONCURRENT_TABS)]

        queue = asyncio.Queue()
        results = {}

        # Populate queue (only idx and url)
        for idx, row in jobs_to_fetch.iterrows():
            url = row.get("job_url")
            if isinstance(url, str) and url.startswith("http"):
                await queue.put((idx, url))

        processed_since_save = 0

        with tqdm(total=len(jobs_to_fetch), desc="Fetching", ncols=90) as pbar:
            # create workers and assign each a dedicated page
            workers = [
                asyncio.create_task(worker(f"W{i}", queue, results, pages[i], pbar))
                for i in range(len(pages))
            ]

            # Periodically autosave progress while queue is being processed
            while not queue.empty():
                await asyncio.sleep(5)
                if len(results) - processed_since_save >= SAVE_INTERVAL:
                    for idx, desc in results.items():
                        df.at[idx, "description_full"] = desc
                    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
                    processed_since_save = len(results)
                    print(f"💾 Progress saved ({len(results)} done).")

            await queue.join()

            # Stop workers by sending sentinel per worker
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers, return_exceptions=True)

        # Final save of results into DataFrame
        for idx, desc in results.items():
            df.at[idx, "description_full"] = desc

        # close pages/context/browser
        for pg in pages:
            try:
                await pg.close()
            except Exception:
                pass
        await context.close()
        await browser.close()

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ Done. Saved final results to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())