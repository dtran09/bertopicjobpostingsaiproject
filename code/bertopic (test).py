import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from wordcloud import WordCloud
from scipy.cluster import hierarchy
import spacy
from umap import UMAP
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import re
import webbrowser

# CSV loader 
def load_csv(path):
	"""
	Tries to detect gzip and attempts several encodings (utf-8, latin-1, cp1252).
	Returns a pandas DataFrame or raises the last exception.
	"""
	# detect gzip magic bytes
	try:
		with open(path, "rb") as f:
			start = f.read(2)
		is_gzip = start == b"\x1f\x8b"
	except Exception:
		is_gzip = False

	compression = "gzip" if is_gzip else None
	encodings = ("utf-8", "latin-1", "cp1252")

	last_exc = None
	for enc in encodings:
		try:
			if compression:
				df = pd.read_csv(path, compression=compression, encoding=enc, engine="python")
			else:
				df = pd.read_csv(path, encoding=enc, engine="python")
			print(f"Loaded {path!r} (compression={compression}, encoding={enc})")
			return df
		except UnicodeDecodeError as e:
			last_exc = e
			# try next encoding
			continue
		except Exception as e:
			# remember and try next encoding / fallback
			last_exc = e
			continue

	# final fallback: force latin-1 (can read any byte sequence)
	try:
		if compression:
			return pd.read_csv(path, compression=compression, encoding="latin-1", engine="python")
		return pd.read_csv(path, encoding="latin-1", engine="python")
	except Exception as e:
		# raise the last meaningful exception
		raise last_exc or e

# detect best text column and return cleaned list of docs
def get_texts_from_df(df, preferred=("text", "description", "job_description", "cleaned_text", "content", "posting", "job_post")):
	"""
	Return (column_name, list_of_texts). Tries preferred names first, then picks
	the object dtype column with largest average string length. Raises ValueError
	if no suitable column is found.
	"""
	if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
		raise ValueError("DataFrame is empty or invalid.")

	# direct hit
	for name in preferred:
		if name in df.columns:
			series = df[name].astype(str).fillna("").astype(str)
			# require some non-empty content
			if series.str.strip().replace("", np.nan).notna().any():
				return name, series.tolist()
			# else continue looking

	# fallback: pick object columns and score by average length
	obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
	if obj_cols:
		best_col = None
		best_len = -1
		for c in obj_cols:
			# compute average non-null string length
			s = df[c].dropna().astype(str)
			if s.empty:
				continue
			avg = s.map(len).mean()
			if avg > best_len:
				best_len = avg
				best_col = c
		if best_col is not None:
			series = df[best_col].astype(str).fillna("").astype(str)
			return best_col, series.tolist()

	# as last resort, consider any column coerced to str with some non-empty values
	for c in df.columns:
		series = df[c].astype(str).fillna("").astype(str)
		if series.str.strip().replace("", np.nan).notna().any():
			return c, series.tolist()

	# nothing suitable
	raise ValueError(f"No text-like column found. Available columns: {list(df.columns)}")

# path for csv file
csv_path = "C:/Users/trand27/Python Projects/Bertopic Test/glassdoor_jobs_cleaned_(2023)_processed.csv"
df = load_csv(csv_path)

# select only 6000 entries
df = df[0:6000]

# load spaCy model with download fallback and provide tokenizer fallback ---
spacy_available = False
nlp = None
try:
	# try to load installed model
	nlp = spacy.load("en_core_web_sm", disable=["ner"])
	spacy_available = True
except Exception:
	try:
		# attempt to download the model and load again
		print("spaCy model 'en_core_web_sm' not found — attempting to download...")
		spacy.cli.download("en_core_web_sm")
		nlp = spacy.load("en_core_web_sm", disable=["ner"])
		spacy_available = True
	except Exception:
		# final fallback: no spaCy model available
		print("Warning: could not load or download 'en_core_web_sm'. Falling back to simple stopword tokenizer.")
		nlp = None
		spacy_available = False

# sklearn stopwords for fallback tokenizer
sklearn_stop = set(ENGLISH_STOP_WORDS)

EXTENDED_STOPWORDS = set(ENGLISH_STOP_WORDS).union({
	"date", "post", "title", "datum", "type", "time", "posted", "posting",
	"apply", "job", "company", "include", "includes", "come", "work", "experience",
	"required", "requirement", "requirements", "role", "position", "opportunity",
	"skill", "skills", "ability", "new", "use", "using", "based", "within",
	"environment", "level", "strong", "excellent", "good", "knowledge", "understanding"
})

def improved_tokenizer(text):
	"""
	More balanced tokenizer that keeps meaningful job-related terms.
	Uses spaCy when available (lemmas + POS filtering), otherwise a regex fallback.
	"""
	if spacy_available and nlp is not None:
		doc = nlp(text or "")
		tokens = []
		for token in doc:
			if (token.is_alpha and
				not token.is_stop and
				token.lemma_.lower() not in EXTENDED_STOPWORDS and
				len(token.lemma_) > 2):
				# Include nouns, proper nouns, adjectives, and some verbs
				if token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}:
					tokens.append(token.lemma_.lower())
		return tokens
	else:
		# Fallback tokenizer: words of length >=3 filtered by extended stopwords
		toks = re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower())
		return [t for t in toks if t not in EXTENDED_STOPWORDS]

# use CountVectorizer with the custom tokenizer to remove adjectives/pronouns and stopwords
# note: token_pattern must be None when using a custom tokenizer
vectorizer_model = CountVectorizer(
	tokenizer=improved_tokenizer,
	token_pattern=None,
	lowercase=True,
	ngram_range=(1, 2),    # include bigrams (captures phrases like "machine learning")
	min_df=0.01,           
	max_df=0.85,           # ignore very frequent terms
	max_features=15000
)

# UMAP for better dimensionality reduction + explicit sentence-transformer embedding
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
model = BERTopic(
	embedding_model="all-MiniLM-L6-v2",   # explicit, compact SBERT model
	umap_model=umap_model,
	vectorizer_model=vectorizer_model,
	min_topic_size=25,
	nr_topics="auto",
	verbose=True
)

# convert to list (replace direct df.text access with robust extractor)
try:
	text_col, docs = get_texts_from_df(df)
	print(f"Using text column: {text_col!r} (n={len(docs)})")
	# drop empty strings if any
	docs = [d for d in docs if d and d.strip()]
	if not docs:
		raise ValueError("No non-empty documents found after extracting the text column.")
except Exception as e:
	raise RuntimeError(f"Failed to extract documents from DataFrame: {e}")

# paragraph splitting helpers (implementing provided snippet) ---
def split_paragraphs_by_newline(records, text_key="Job Description"):
	"""
	For each record (dict) in records, split the text_key value by single newline,
	store list under record["paragraphs"], and return the flattened list of paragraphs.
	Accepts a pandas DataFrame (converted to records) or list-of-dicts.
	"""
	if isinstance(records, pd.DataFrame):
		records = records.to_dict(orient="records")
	paras_all = []
	for r in records:
		jb = r.get(text_key, "") if isinstance(r, dict) else ""
		paras = [p.strip() for p in str(jb).split("\n") if p.strip()]
		r["paragraphs"] = paras
		paras_all.extend(paras)
	return paras_all

def split_paragraphs_by_two_newline(records, text_key="Job Description"):
	"""
	Similar to split_paragraphs_by_newline but splits on double-newline ("\n\n").
	"""
	if isinstance(records, pd.DataFrame):
		records = records.to_dict(orient="records")
	paras_all = []
	for r in records:
		jb = r.get(text_key, "") if isinstance(r, dict) else ""
		paras = [p.strip() for p in str(jb).split("\n\n") if p.strip()]
		r["paragraphs"] = paras
		paras_all.extend(paras)
	return paras_all

# Use the detected text_col from get_texts_from_df as the key when possible.
# Convert df to records and build both paragraph lists for later use.
try:
	df_records = df.to_dict(orient="records")
	paragraphs_by_newline = split_paragraphs_by_newline(df_records, text_key=text_col)
	paragraphs_by_two_newline = split_paragraphs_by_two_newline(df_records, text_key=text_col)
	# optional: also expose paragraph lists derived directly from the docs list
	# (docs are plain strings; convert temporarily)
	docs_records = [{text_col: d} for d in docs]
	docs_paragraphs_by_newline = split_paragraphs_by_newline(docs_records, text_key=text_col)
	docs_paragraphs_by_two_newline = split_paragraphs_by_two_newline(docs_records, text_key=text_col)
except Exception as e:
	print("Warning: failed to generate paragraph splits:", e)

# Fit model and try to obtain per-document probabilities in a backward/forward-compatible way
try:
	topics, probabilities = model.fit_transform(docs, calculate_probabilities=True)
except TypeError:
	# fit_transform doesn't accept calculate_probabilities for this BERTopic version
	res = model.fit_transform(docs)
	if isinstance(res, tuple) and len(res) == 2:
		topics, probabilities = res
	else:
		# fit_transform returned only topics
		topics = res
		probabilities = None
		# try to get probabilities via transform() — many BERTopic versions return (topics, probs)
		try:
			res2 = model.transform(docs)
			if isinstance(res2, tuple) and len(res2) == 2:
				topics, probabilities = res2
		except Exception:
			# try common alternative kwarg names used across versions
			for kw in ("return_probabilities", "probabilities", "return_proba"):
				try:
					res2 = model.transform(docs, **{kw: True})
					if isinstance(res2, tuple) and len(res2) == 2:
						topics, probabilities = res2
						break
				except Exception:
					pass

# get topics and probabilities
topic_freq = model.get_topic_freq()             # assign so we can inspect / reuse
print("Top 10 topic frequencies:\n", topic_freq.head(10))

# Print a single topic so you can inspect its words (if it exists)
topic = model.get_topic(9)
print("Topicwords:", topic)

# visualize topics as small barcharts like the screenshot
fig = model.visualize_barchart(top_n_topics=10, n_words=5)
fig.show()

def get_top_word_probs(model, topic_id, n=8, method="normalize"):  # Reduced n from 10 to 8
    """
    Improved topic word selection with better filtering
    """
    topic = model.get_topic(topic_id)
    if not topic:
        return []
    
    # Filter out less meaningful terms and duplicates
    filtered = []
    seen_lemmas = set()
    
    for word, score in topic:
        word_lower = word.lower()
        # Skip if in extended stopwords or already seen
        if (word_lower not in EXTENDED_STOPWORDS and 
            word_lower not in seen_lemmas and
            len(word) > 2):
            filtered.append((word, score))
            seen_lemmas.add(word_lower)
    
    if not filtered:
        return []
    
    # Take top n words
    words, scores = zip(*filtered[:n])
    scores = np.array(scores, dtype=float)
    
    # Normalize probabilities
    if method == "softmax":
        exps = np.exp(scores - scores.max())
        probs = exps / exps.sum()
    else:  # normalize
        s = scores.sum() if scores.sum() != 0 else 1.0
        probs = scores / s
        
    return [{"word": w, "score": float(s), "prob": float(p)} for w, s, p in zip(words, scores, probs)]

def get_all_topics_top_word_probs(model, n=10, method="normalize", skip_negative=True):
	"""
	Returns dict {topic_id: [word_probs...], ...}
	"""
	info = model.get_topic_info()
	topics = {}
	for tid in info.Topic.tolist():
		if skip_negative and tid == -1:
			continue
		topics[int(tid)] = get_top_word_probs(model, int(tid), n=n, method=method)
	return topics

def plot_topic_word_probs(topic_id, word_probs, show=True, save_html=None):
	"""
	Plot horizontal bar chart of words vs probability using Plotly.
	Returns the figure.
	"""
	if not word_probs:
		raise ValueError(f"No words for topic {topic_id}")
	words = [w["word"] for w in word_probs]
	probs = [w["prob"] for w in word_probs]
	fig = go.Figure(go.Bar(x=probs[::-1], y=words[::-1], orientation="h",
						   text=[f"{p:.4f}" for p in probs[::-1]], textposition="auto"))
	fig.update_layout(title=f"Topic {topic_id} top words (probabilities)", xaxis_title="Probability")
	if save_html:
		fig.write_html(save_html)
	if show:
		fig.show()
	return fig

# Build probabilities for all topics (top 10 words by default)
all_topic_word_probs = get_all_topics_top_word_probs(model, n=10, method="normalize")

# plot topic if present
if 6 in all_topic_word_probs:
	try:
		plot_topic_word_probs(6, all_topic_word_probs[6])
	except Exception as e:
		print(f"Could not plot topic 10: {e}")

def summarize_topic_confidence(topics_list, probs_list):
	"""
	Compute per-topic document count and mean confidence (uses max prob per doc).
	Returns dict {topic: {"count": int, "mean_conf": float}}
	"""
	stats = {}
	if probs_list is None:
		return stats
	for tid, p in zip(topics_list, probs_list):
		# determine a single confidence value for this doc
		conf = 0.0
		if p is None:
			conf = 0.0
		elif isinstance(p, (list, np.ndarray)):
			arr = np.array(p, dtype=float)
			if arr.size == 0:
				conf = 0.0
			elif arr.ndim == 0 or arr.size == 1:
				conf = float(arr)
			else:
				conf = float(arr.max())  # use max probability as the doc's confidence
		else:
			try:
				conf = float(p)
			except Exception:
				conf = 0.0
		entry = stats.setdefault(tid, {"count": 0, "sum_conf": 0.0})
		entry["count"] += 1
		entry["sum_conf"] += conf
	# finalize mean
	for tid, v in stats.items():
		v["mean_conf"] = v["sum_conf"] / v["count"] if v["count"] else 0.0
	return stats

topic_conf_stats = summarize_topic_confidence(topics, probabilities)
# Print top topics by mean confidence (highest first)
sorted_stats = sorted(((tid, v["count"], v["mean_conf"]) for tid, v in topic_conf_stats.items()),
					  key=lambda x: x[2], reverse=True)
print("Topic confidence (topic, count, mean_conf) top 10 by mean_conf:")
for row in sorted_stats[:10]:
	print(row)

def top_docs_for_topic(topic_id, topics_list, probs_list, documents, n=12):
	"""
	Return up to n documents assigned to topic_id, sorted by confidence (descending).
	"""
	indices = [i for i, t in enumerate(topics_list) if t == topic_id]
	if not indices:
		return []
	rank = []
	for i in indices:
		p = probs_list[i] if probs_list is not None else None
		# derive a single confidence like above
		if p is None:
			conf = 0.0
		elif isinstance(p, (list, np.ndarray)):
			arr = np.array(p, dtype=float)
			conf = float(arr.max()) if arr.size else 0.0
		else:
			try:
				conf = float(p)
			except Exception:
				conf = 0.0
		rank.append((i, conf))
	rank.sort(key=lambda x: x[1], reverse=True)
	return [(i, round(conf, 4), documents[i]) for i, conf in rank[:n]]

# show top documents for topic (if present)
if 9 in topic_conf_stats:
	top_docs = top_docs_for_topic(9, topics, probabilities, docs, n=5)
	print(f"Top docs for topic 9 (index, conf, snippet):")
	for idx, conf, doc in top_docs:
		print(idx, conf, doc[:200].replace("\n", " "))

# build full topic-probability matrix (n_docs x n_topics) and return topic order
def build_full_topic_matrix(assigned_topics, probs, model, skip_negative=True):
	"""
	Returns (full_matrix, topic_order)
	full_matrix: numpy array shape (n_docs, n_topics) where columns follow topic_order
	topic_order: list of topic ids corresponding to columns
	Handles multiple possible shapes of `probs` returned by BERTopic.
	"""
	info = model.get_topic_info()
	topic_order = [int(t) for t in info.Topic.tolist() if not (skip_negative and int(t) == -1)]
	n_docs = len(assigned_topics)
	n_topics = len(topic_order)
	full = np.zeros((n_docs, n_topics), dtype=float)

	if probs is None:
		return full, topic_order

	# Case: full 2D numpy array already
	if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[0] == n_docs:
		# If shapes match number of topics, assume it's already full matrix
		if probs.shape[1] == n_topics:
			return probs, topic_order
		# If columns mismatch, fall back to per-doc parsing below

	# Per-document parsing (handle list of tuples, list of floats, scalar, etc.)
	for i, p in enumerate(probs):
		if p is None:
			continue

		# Helper: assign a value to the assigned topic column (if possible)
		def assign_to_assigned(val):
			try:
				assigned = int(assigned_topics[i]) if assigned_topics is not None else None
				if assigned is None:
					return
				if assigned == -1 and skip_negative:
					return
				idx = topic_order.index(int(assigned))
				full[i, idx] = float(val)
			except Exception:
				# ignore assignment errors
				pass

		# Numpy scalar (0-d) or plain python scalar
		if np.isscalar(p) or (isinstance(p, np.ndarray) and p.ndim == 0):
			try:
				assign_to_assigned(float(p))
			except Exception:
				continue
			continue

		# Convert to list-like for inspection where safe
		try:
			seq = list(p) if not isinstance(p, (str, bytes)) else [p]
		except Exception:
			# fallback: treat as scalar assigned to topic
			assign_to_assigned(p)
			continue

		# Now seq is a sequence (possibly of pairs or numbers)
		if len(seq) == 0:
			continue

		# Case: sequence of (topic, prob) pairs
		first = seq[0]
		if isinstance(first, (list, tuple)) and len(first) == 2:
			for pair in seq:
				try:
					tid, pr = pair
					if skip_negative and int(tid) == -1:
						continue
					try:
						idx = topic_order.index(int(tid))
						full[i, idx] = float(pr)
					except ValueError:
						# topic id not in current topic_order -> ignore
						continue
				except Exception:
					continue
			continue

		# Case: numeric vector matching n_topics
		# try to convert seq to numeric array
		try:
			num = np.array(seq, dtype=float)
			if num.ndim == 1 and num.size == n_topics:
				full[i, :] = num
				continue
			# if 1D but size differs, fall through to assign representative value
		except Exception:
			# not a pure numeric sequence
			pass

		# Fallback: assign a representative confidence to the assigned topic
		# try max numeric value in seq, otherwise use first element if numeric
		numeric_vals = []
		for el in seq:
			try:
				numeric_vals.append(float(el))
			except Exception:
				continue
		if numeric_vals:
			assign_to_assigned(max(numeric_vals))
		else:
			# give up on this doc's probabilities
			continue

	return full, topic_order

# Build the full distribution matrix for the original model
full_matrix, topic_order = build_full_topic_matrix(topics, probabilities, model)
# Create a DataFrame for easier inspection: columns named topic_{id}
col_names = [f"topic_{tid}" for tid in topic_order]
full_df = pd.DataFrame(full_matrix, columns=col_names)
full_df.insert(0, "assigned_topic", topics)
full_df.insert(0, "document", docs)

# Show/save an example
print("Per-document topic-distribution sample (first 5 rows):")
print(full_df.head(5).to_string(index=False))
full_df.to_csv("c:/Users/trand27/Python Projects/Bertopic Test/doc_topic_distributions_full.csv", index=False)

def save_topic_pages(topic_word_probs_dict, out_dir):
	"""
	Save one HTML page per topic showing the Plotly bar chart for that topic.
	Also generate one combined wordcloud for all topics and write an HTML page for it.
	topic_word_probs_dict: {topic_id: [ {"word","score","prob"}, ... ] }
	out_dir: directory to place topic_{id}.html and all_topics_wordcloud.png/html
	"""
	os.makedirs(out_dir, exist_ok=True)

	# --- Build combined frequency map across all topics ---
	combined_freqs = {}
	for tid, word_probs in topic_word_probs_dict.items():
		for w in word_probs:
			try:
				weight = w.get("prob")
				if weight is None:
					weight = w.get("score", 1.0)
				combined_freqs[str(w["word"])] = combined_freqs.get(str(w["word"]), 0.0) + float(weight)
			except Exception:
				continue

	# Create one big wordcloud image for all topics
	try:
		if combined_freqs:
			combined_img = os.path.join(out_dir, "all_topics_wordcloud.png")
			wc = WordCloud(width=1400, height=700, background_color="white")
			wc.generate_from_frequencies(combined_freqs)
			wc.to_file(combined_img)

			# Create an HTML page that embeds the combined wordcloud image
			combined_html = os.path.join(out_dir, "all_topics_wordcloud.html")
			page_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8' />
<title>All Topics — Combined Wordcloud</title>
</head>
<body>
<h1>All Topics Combined Wordcloud</h1>
<div><img src="all_topics_wordcloud.png" alt="Combined wordcloud" style="max-width:100%;height:auto;"/></div>
</body>
</html>"""
			with open(combined_html, "w", encoding="utf-8") as f:
				f.write(page_html)
		else:
			combined_html = None
			print("Warning: combined word frequencies empty; not creating combined wordcloud.")
	except Exception as e:
		combined_html = None
		print("Warning: could not create combined wordcloud:", e)

	# --- Per-topic pages (Plotly chart only) ---
	for tid, word_probs in topic_word_probs_dict.items():
		try:
			# skip empty topics
			if not word_probs:
				print(f"Warning: no words for topic {tid}, skipping page.")
				continue

			# create the Plotly figure for the topic (do not write standalone HTML)
			fig = plot_topic_word_probs(int(tid), word_probs, show=False, save_html=None)

			# embed the plotly fragment in a single HTML page for the topic
			div_id = f"topic_plot_{int(tid)}"
			plot_fragment = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=div_id)

			page_path = os.path.join(out_dir, f"topic_{int(tid)}.html")
			page_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8' />
<title>Topic {int(tid)} — words</title>
</head>
<body>
<h1>Topic {int(tid)}</h1>
<div>
{plot_fragment}
</div>
</body>
</html>"""
			with open(page_path, "w", encoding="utf-8") as f:
				f.write(page_html)

		except Exception as e:
			# skip problematic topics but continue
			print(f"Warning: could not write page for topic {tid}: {e}")

	# If we created a combined page, open it in the default browser once
	try:
		if combined_html:
			webbrowser.open("file://" + os.path.abspath(combined_html))
			print(f"Wrote and opened combined wordcloud page: {combined_html}")
	except Exception as e:
		print("Warning: could not open combined wordcloud in browser:", e)

def save_intertopic_with_clicks(fig, out_path, topics_rel_dir="topic_pages"):
	"""
	Save an intertopic map HTML with a small JS handler:
	- on click, parse the clicked point's text/hovertext/customdata for a topic id
	- open topic_pages/topic_{id}.html and topic_pages/all_topics_wordcloud.html in new tabs
	"""
	# Ensure the div id is stable so our JS can attach to it
	div_id = "intertopic_map"
	fig_html_fragment = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=div_id)

	# JS: attach plotly_click handler and try to extract a topic id from multiple parts
	# This version opens both the topic page and the combined wordcloud page when clicking
	post_script = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
  var gd = document.getElementById('{div_id}');
  if(!gd) return;
  gd.on('plotly_click', function(data) {{
    try {{
      var p = (data && data.points && data.points[0]) || {{}};
      var candidates = [];

      if(p.text) candidates.push(p.text);
      if(p.hovertext) candidates.push(p.hovertext);
      if(p.customdata) candidates.push(p.customdata);
      if(p.customdata && p.customdata[0]) candidates.push(p.customdata[0]);
      if(p.data && p.data.name) candidates.push(p.data.name);
      if(typeof p.pointIndex !== 'undefined') candidates.push(p.pointIndex.toString());
      if(typeof p.pointNumber !== 'undefined') candidates.push(p.pointNumber.toString());
      if(typeof p.curveNumber !== 'undefined') candidates.push(p.curveNumber.toString());

      try {{ candidates.push(JSON.stringify(p)); }} catch(e) {{ /* ignore */ }}

      var tid = null;
      for(var i=0;i<candidates.length;i++) {{
        var txt = candidates[i];
        if(typeof txt === 'object') {{
          try {{ txt = JSON.stringify(txt); }} catch(e) {{ txt = ''; }}
        }}
        if(!txt) continue;
        var m = txt.toString().match(/-?\\d+/);
        if(m) {{ tid = m[0]; break; }}
      }}
      if(tid !== null) {{
        var topicUrl = '{topics_rel_dir}/topic_' + tid + '.html';
        var wcUrl = '{topics_rel_dir}/all_topics_wordcloud.html';
        // open topic page and combined wordcloud page in separate tabs
        window.open(topicUrl, '_blank');
        window.open(wcUrl, '_blank');
        return;
      }}
    }} catch(e) {{
      console.log('click handler error', e);
    }}
  }});
}});
</script>
"""
	# Build full html
	full_html = f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8' />\n<title>Intertopic Map (click a topic)</title>\n</head>\n<body>\n{fig_html_fragment}\n{post_script}\n</body>\n</html>"
	# write out
	with open(out_path, "w", encoding="utf-8") as f:
		f.write(full_html)
	print(f"Wrote intertopic map with click handler to: {out_path}")

# create topic pages directory (relative to the intertopic HTML) and write files
output_base = r"c:\Users\trand27\Python Projects\Bertopic Test"
topic_pages_dir = os.path.join(output_base, "topic_pages")
save_topic_pages(all_topic_word_probs, topic_pages_dir)

# save intertopic map HTML (the fig variable used earlier - reuse same figure)
intertopic_html_path = os.path.join(output_base, "intertopic_map_clickable.html")
try:
	save_intertopic_with_clicks(fig, intertopic_html_path, topics_rel_dir="topic_pages")
except Exception as e:
	print("Could not save clickable intertopic HTML:", e)

# Save BERTopic model
model_save_base = os.path.join(output_base, "bertopic_model")
try:
	# BERTopic.save typically writes a directory or file at the given path
	model.save(model_save_base)
	print(f"Saved BERTopic model to: {model_save_base}")
except Exception as e_save:
	try:
		import pickle
		pkl_path = model_save_base + ".pkl"
		with open(pkl_path, "wb") as f:
			pickle.dump(model, f)
		print(f"BERTopic.save failed ({e_save!r}), model pickled to: {pkl_path}")
	except Exception as e_pickle:
		print("Failed to save BERTopic model using both model.save and pickle:", e_save, e_pickle)

# Generate and show hierarchical clustering of topics, then save as HTML
try:
	# linkage using scipy.cluster.hierarchy (optimal_ordering can help layout)
	linkage_function = lambda x: hierarchy.linkage(x, method="single", optimal_ordering=True)
	hierarchical_topics = model.hierarchical_topics(docs, linkage_function=linkage_function)
	fig_hierarchy = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
	# display and save
	fig_hierarchy.show()
	hier_html = os.path.join(output_base, "hierarchy_map.html")
	try:
		fig_hierarchy.write_html(hier_html, include_plotlyjs="cdn")
		print(f"Wrote hierarchical topics visualization to: {hier_html}")
	except Exception as e_w:
		# fallback: try to_html and write manually
		try:
			fragment = fig_hierarchy.to_html(full_html=True, include_plotlyjs="cdn")
			with open(hier_html, "w", encoding="utf-8") as fh:
				fh.write(fragment)
			print(f"Wrote hierarchical topics visualization to: {hier_html}")
		except Exception as e2:
			print("Failed to save hierarchy HTML:", e_w, e2)
except Exception as e:
	print("Could not generate/save hierarchical topics visualization:", e)