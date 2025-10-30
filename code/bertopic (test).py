import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go

# robust CSV loader (already present)
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

# new helper: detect best text column and return cleaned list of docs
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

# use CountVectorizer to remove English stopwords from the topic word extraction
vectorizer_model = CountVectorizer(stop_words="english")
model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)

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

# visualize topics (fixed call)
fig = model.visualize_topics(top_n_topics=20)
###### fig = model.visualize_barchart()
###### fig = model.visualize_heatmap()
fig.show()

def get_top_word_probs(model, topic_id, n=10, method="normalize"):
	"""
	Returns list of dicts: [{"word": str, "score": float, "prob": float}, ...]
	method: "normalize" (score / sum(scores)) or "softmax"
	"""
	topic = model.get_topic(topic_id)
	if not topic:
		return []
	words, scores = zip(*topic[:n])
	scores = np.array(scores, dtype=float)
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

# Example: plot topic 10 if present
if 5 in all_topic_word_probs:
	try:
		plot_topic_word_probs(5, all_topic_word_probs[5])
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

# Example: show top documents for topic (if present)
if 9 in topic_conf_stats:
	top_docs = top_docs_for_topic(9, topics, probabilities, docs, n=5)
	print(f"Top docs for topic 9 (index, conf, snippet):")
	for idx, conf, doc in top_docs:
		print(idx, conf, doc[:200].replace("\n", " "))
# --- end probability analysis ---------------------------------------------

# helper: build full topic-probability matrix (n_docs x n_topics) and return topic order
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