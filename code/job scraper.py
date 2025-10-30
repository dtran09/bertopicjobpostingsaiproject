import csv
from jobspy import scrape_jobs
import re
import pandas as pd
from html import unescape

jobs = scrape_jobs(
    site_name=["glassdoor"], # "glassdoor", "bayt", "naukri", "bdjobs"
    search_term="data science",
    location="Boston",
    results_wanted=300,
    country_indeed='USA',
    
    linkedin_fetch_description=True,  # <-- enable full description/direct url (may be slower)
    # proxies=["208.195.175.46:65095", "208.195.175.45:65095", "localhost"],
)

print(f"Found {len(jobs)} jobs")
print(jobs.head())

# --- existing keyword maps for job_type / job_level / job_function ---
job_type_map = {
    "full-time": ["full-time", "full time", "permanent", "fte"],
    "part-time": ["part-time", "part time"],
    "contract": ["contract", "contractor", "temporary", "temp"],
    "internship": ["intern", "internship"],
    "hourly": ["hourly", "per hour", "/hour"],
}
job_level_map = {
    "senior": ["senior", "sr.", "sr ", "sr-"],
    "junior": ["junior", "jr.", "jr ", "jr-"],
    "lead": ["lead ", "lead,", "team lead", "technical lead"],
    "manager": ["manager", "mgr", "management"],
    "director": ["director"],
    "principal": ["principal"],
    "entry-level": ["entry level", "entry-level"],
    "associate": ["associate"],
    "mid-level": ["mid level", "mid-level"],
    "intern": ["intern"],
}
job_function_map = {
    "engineering": ["engineer", "engineering", "devops", "software", "developer", "full stack", "backend", "frontend"],
    "it": ["it ", "information technology", "system admin", "system administrator", "system engineer"],
    "data": ["data scientist", "data engineer", "data analyst", "data"],
    "support": ["support", "help desk", "technical support", "desktop support"],
    "product": ["product manager", "product"],
    "sales": ["sales", "account executive", "business development"],
    "hr": ["human resources", "hr "],
    "marketing": ["marketing"],
    "finance": ["finance", "financial", "accounting"],
    "qa": ["qa", "quality assurance", "test engineer"],
    "design": ["designer", "ux", "ui", "product designer"],
    "customer success": ["customer success", "customer service"],
    "security": ["security", "cyber", "info sec", "infosec"],
}

def find_keyword(text: str, mapping: dict):
    if not text:
        return None
    tl = text.lower()
    for label, variants in mapping.items():
        for v in variants:
            if v in tl:
                return label
    return None

# --- ensure all target columns exist ---
target_cols = (
    "job_type", "job_level", "job_function",
    "emails", "description", "company_industry", "company_url_direct", "company_addresses",
    "company_num_employees", "company_revenue", "company_description", "skills",
    "experience_range", "company_rating", "company_reviews_count", "vacancy_count", "work_from_home_type"
)
for col in target_cols:
    if col not in jobs.columns:
        jobs[col] = ""

# combine title + description + any summary fields that might exist
def combined_text(row):
    parts = []
    for k in ("title", "description", "job_summary", "job_title", "snippet", "summary", "job_description", "description_text", "details", "full_description", "linkedin_description"):
        if k in row and pd.notna(row.get(k)):
            parts.append(strip_html(str(row.get(k))))
    # also include any company text columns
    for cname in row.index:
        if isinstance(cname, str) and "company" in cname.lower():
            val = row.get(cname)
            if pd.notna(val) and str(val).strip():
                parts.append(strip_html(str(val)))
    return " ".join(parts)

# extract emails from text
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', flags=re.I)
def extract_emails(text):
    if not text:
        return ""
    found = set(EMAIL_RE.findall(text))
    return ", ".join(sorted(found))

# find numeric pattern for experience like "2-4 years" or "3+ years"
EXP_RE = re.compile(r'(\d+)\s*(?:\+|-|to|–)?\s*(\d+)?\s*(?:years|yrs|year)', flags=re.I)
def extract_experience(text):
    if not text:
        return ""
    m = EXP_RE.search(text)
    if m:
        a = m.group(1)
        b = m.group(2)
        if b:
            return f"{a}-{b} years"
        if "+" in m.group(0) or re.search(r'\d+\s*\+', m.group(0)):
            return f"{a}+ years"
        return f"{a} years"
    # fallback: "entry level", "senior" etc
    if re.search(r'entry[- ]level', text, flags=re.I):
        return "entry-level"
    if re.search(r'\bjunior\b|\bjr\b', text, flags=re.I):
        return "junior"
    if re.search(r'\bsenior\b|\bsr\b', text, flags=re.I):
        return "senior"
    return ""

# extract skills by presence of common keywords
COMMON_SKILLS = ["python","java","sql","aws","azure","gcp","linux","docker","kubernetes","react","angular","node","javascript","c#","c++","git","nosql","mongodb","pandas","spark","hadoop","tableau"]
def extract_skills(text):
    if not text:
        return ""
    tl = text.lower()
    found = [s for s in COMMON_SKILLS if (" " + s) in (" " + tl) or s in tl]
    return ", ".join(sorted(set(found)))

# find first non-empty company-related column by substring
def find_company_field(row, substrings):
    for col in row.index:
        lname = str(col).lower()
        for s in substrings:
            if s in lname:
                val = row.get(col)
                if pd.notna(val) and str(val).strip() != "":
                    return str(val)
    return ""

# try to extract numeric counts from text (reviews, vacancies)
def extract_number_near_word(text, word):
    if not text:
        return ""
    # look for patterns like "123 reviews" or "Reviews: 123"
    m = re.search(r'(\d{1,6})\s*(?:\+)?\s*(?:' + re.escape(word) + r')', text, flags=re.I)
    if m:
        return m.group(1)
    # fallback: "rating 4.5 (123 reviews)"
    m2 = re.search(r'\((\d{1,6})\s+reviews\)', text, flags=re.I)
    if m2:
        return m2.group(1)
    return ""

# add helper to strip HTML and unescape entities
def strip_html(text):
    if not text:
        return ""
    # remove script/style blocks
    text = re.sub(r'(?is)<(script|style).*?>.*?</\1>', ' ', text)
    # strip tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return unescape(text).strip()

# new helper: detect image/logo-like content or URL-only values
def is_image_like(text, colname=None):
    if not text:
        return False
    t = str(text).strip()
    tl = t.lower()
    # column name hints
    if colname:
        cname = str(colname).lower()
        if any(x in cname for x in ("logo", "image", "icon", "thumbnail", "avatar", "photo", "picture")):
            return True
    # HTML image tags or svg
    if re.search(r'<\s*img\b|<\s*svg\b|data:image\/|base64,', tl):
        return True
    # src= attribute or <picture> tags
    if 'src=' in tl or 'src :' in tl:
        return True
    # URL that ends with common image extension (possibly with query)
    if re.search(r'https?://\S+\.(png|jpg|jpeg|gif|svg|webp)(\?\S*)?$', tl):
        return True
    # single token that is an http(s) url or file path with image ext
    if (tl.startswith('http://') or tl.startswith('https://') or tl.startswith('www.')) and len(t.split()) == 1:
        if re.search(r'\.(png|jpg|jpeg|gif|svg|webp)(\?\S*)?$', tl) or 'logo' in tl:
            return True
    # short values dominated by punctuation/urls (likely not a description)
    words = re.findall(r'\w+', t)
    if len(words) < 8 and (('http' in tl) or re.search(r'\.\w{2,4}$', tl)):
        return True
    return False

# modified helper: pick best description candidate from row (handles bytes/lists/dicts and HTML-like content)
def find_description_candidate(row):
    def norm_val(v, colname=None):
        if v is None:
            return ""
        # handle bytes
        if isinstance(v, bytes):
            try:
                v = v.decode('utf-8', errors='ignore')
            except Exception:
                v = str(v)
        # flatten lists/tuples/sets and filter image-like items
        if isinstance(v, (list, tuple, set)):
            parts = []
            for x in v:
                sx = "" if x is None else str(x)
                if not is_image_like(sx):
                    parts.append(sx)
            return " ".join(parts)
        # flatten dict by values and filter image-like items
        if isinstance(v, dict):
            parts = []
            for x in v.values():
                sx = "" if x is None else str(x)
                if not is_image_like(sx):
                    parts.append(sx)
            return " ".join(parts)
        return str(v)

    candidates = (
        "description", "job_description", "job_summary", "snippet", "summary",
        "description_text", "details", "full_description", "linkedin_description",
        "job_details", "raw_description", "description_html", "job_html", "posting_text",
        "body", "content", "ad_text", "job_posting"
    )
    best = ""
    # check prioritized keys first, skip columns that look like logos/images
    for k in candidates:
        if k in row and pd.notna(row.get(k)):
            v_raw = row.get(k)
            # skip if the raw value appears image-like or the column name suggests image
            if is_image_like(v_raw, colname=k):
                continue
            v = norm_val(v_raw, colname=k).strip()
            if not v:
                continue
            # prefer HTML-like or long textual content
            if re.search(r'<\/?\w+[^>]*>|&lt;|&gt;|<p|<div|br\s*\/?>', v, flags=re.I) or len(v) > 120 or re.search(r'\b(responsibilit|qualification|requirement|role|about the company)\b', v, flags=re.I):
                return v
            # keep as potential fallback (longest non-image candidate)
            if len(v) > len(best):
                best = v
    # if none of the prioritized keys produced HTML-like or long, scan all text columns and pick the longest
    if not best:
        longest = ""
        for col in row.index:
            try:
                val = row.get(col)
            except Exception:
                continue
            if pd.isna(val) or val is None:
                continue
            # skip columns whose name suggests images
            if isinstance(col, str) and any(x in col.lower() for x in ("logo", "image", "icon", "thumbnail", "avatar", "photo", "picture")):
                continue
            # skip values that are image-like
            if is_image_like(val, colname=col):
                continue
            v = norm_val(val, colname=col).strip()
            if not v:
                continue
            # prefer longer substantial text
            if len(v) > len(longest) and len(v) > 80:
                longest = v
        if longest:
            return longest
    return best

# apply filling only where empty
def fill_row(row):
    # attempt to populate description early so combined_text() can include it
    if (not row.get("description")) or pd.isna(row.get("description")) or str(row.get("description")).strip()=="":
        desc_candidate = find_description_candidate(row)
        if desc_candidate:
            row["description"] = strip_html(desc_candidate)

    # recompute combined text now that description may be set
    text = combined_text(row)

    # job_type / job_level / job_function (existing behavior)
    if (not row.get("job_type")) or pd.isna(row.get("job_type")) or str(row.get("job_type")).strip()=="":
        interval = row.get("interval", "")
        if pd.notna(interval) and str(interval).strip() != "":
            row["job_type"] = str(interval)
        else:
            found = find_keyword(text, job_type_map)
            if found:
                row["job_type"] = found
    if (not row.get("job_level")) or pd.isna(row.get("job_level")) or str(row.get("job_level")).strip()=="":
        found = find_keyword(text, job_level_map)
        if found:
            row["job_level"] = found
    if (not row.get("job_function")) or pd.isna(row.get("job_function")) or str(row.get("job_function")).strip()=="":
        found = find_keyword(text, job_function_map)
        if found:
            row["job_function"] = found

    # emails
    if (not row.get("emails")) or pd.isna(row.get("emails")) or str(row.get("emails")).strip()=="":
        emails = extract_emails(text)
        row["emails"] = emails

    # company fields: use best matching company column if present
    if (not row.get("company_industry")) or pd.isna(row.get("company_industry")) or str(row.get("company_industry")).strip()=="":
        row["company_industry"] = find_company_field(row, ("industry", "sector", "field"))

    if (not row.get("company_url_direct")) or pd.isna(row.get("company_url_direct")) or str(row.get("company_url_direct")).strip()=="":
        # look for direct url columns or a 'company_url' like column
        row["company_url_direct"] = find_company_field(row, ("url", "website", "company_url", "direct"))

    if (not row.get("company_addresses")) or pd.isna(row.get("company_addresses")) or str(row.get("company_addresses")).strip()=="":
        # try location fields and any company address columns
        addr = find_company_field(row, ("address", "location", "office"))
        if not addr:
            addr = row.get("location", "")
            if pd.notna(addr):
                addr = str(addr)
        row["company_addresses"] = addr

    if (not row.get("company_num_employees")) or pd.isna(row.get("company_num_employees")) or str(row.get("company_num_employees")).strip()=="":
        row["company_num_employees"] = find_company_field(row, ("employees", "size", "num_employees", "company_size"))

    if (not row.get("company_revenue")) or pd.isna(row.get("company_revenue")) or str(row.get("company_revenue")).strip()=="":
        row["company_revenue"] = find_company_field(row, ("revenue", "turnover", "income"))

    if (not row.get("company_description")) or pd.isna(row.get("company_description")) or str(row.get("company_description")).strip()=="":
        row["company_description"] = find_company_field(row, ("description", "about", "overview"))

    # skills
    if (not row.get("skills")) or pd.isna(row.get("skills")) or str(row.get("skills")).strip()=="":
        row["skills"] = extract_skills(text)

    # experience_range
    if (not row.get("experience_range")) or pd.isna(row.get("experience_range")) or str(row.get("experience_range")).strip()=="":
        exp = extract_experience(text)
        row["experience_range"] = exp

    # company_rating / reviews_count
    if (not row.get("company_rating")) or pd.isna(row.get("company_rating")) or str(row.get("company_rating")).strip()=="":
        # try company columns with rating-like names
        rating = find_company_field(row, ("rating", "score", "company_rating", "overall_rating", "glassdoor_rating", "employer_rating"))
        if rating:
            row["company_rating"] = str(rating)
        else:
            # try to extract a float like 4.3 from text (e.g., "4.3 out of 5" or "rating: 4.3")
            m = re.search(r'(?<!\d)([1-5](?:\.\d)?)\s*(?:/5|out of 5)?', combined_text(row), flags=re.I)
            if m:
                row["company_rating"] = m.group(1)
    if (not row.get("company_reviews_count")) or pd.isna(row.get("company_reviews_count")) or str(row.get("company_reviews_count")).strip()=="":
        rv = find_company_field(row, ("reviews", "review_count"))
        if rv:
            row["company_reviews_count"] = rv
        else:
            extracted = extract_number_near_word(text, "reviews")
            row["company_reviews_count"] = extracted

    # vacancy_count
    if (not row.get("vacancy_count")) or pd.isna(row.get("vacancy_count")) or str(row.get("vacancy_count")).strip()=="":
        vac = find_company_field(row, ("vacanc", "openings", "jobs_open"))
        if vac:
            row["vacancy_count"] = vac
        else:
            row["vacancy_count"] = extract_number_near_word(text, "vacanc")

    # work_from_home_type
    if (not row.get("work_from_home_type")) or pd.isna(row.get("work_from_home_type")) or str(row.get("work_from_home_type")).strip()=="":
        if re.search(r'\bremote\b', text, flags=re.I):
            row["work_from_home_type"] = "remote"
        elif re.search(r'\bhybrid\b', text, flags=re.I):
            row["work_from_home_type"] = "hybrid"
        elif re.search(r'\bon[- ]?site\b|\bonsite\b', text, flags=re.I):
            row["work_from_home_type"] = "on-site"

    return row

jobs = jobs.apply(fill_row, axis=1)

# write CSV with BOM for Excel
jobs.to_csv("jobs_datascience.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False, encoding='utf-8-sig') # to_excel