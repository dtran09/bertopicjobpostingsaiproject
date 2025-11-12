import argparse
from pathlib import Path
import pandas as pd
import sys


input_path = Path("C:/Users/trand27/Python Projects/Bertopic Test/glassdoor_jobs_cleaned_(2023)_processed.csv")
output_path = Path("C:/Users/trand27/Python Projects/Bertopic Test/glassdoor_jobs_cleaned_(2023)_unique.csv")

def dedupe_job_descriptions(input_path: Path, output_path: Path, sheet_name=None, col_name='Job Description'):
	# Read CSV or Excel depending on input file extension
	if input_path.suffix.lower() in ('.xls', '.xlsx'):
		# sheet_name may be None, int or string; pandas handles it
		# preserve literal "N/A" strings (don't treat them as NaN)
		df = pd.read_excel(input_path, sheet_name=sheet_name if sheet_name is not None else 0, keep_default_na=False)
	else:
		# Try common encodings to avoid UnicodeDecodeError on Windows CSVs
		encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
		df = None
		used_encoding = None
		last_exc = None
		for enc in encodings:
			try:
				# preserve literal "N/A" strings (don't treat them as NaN)
				df = pd.read_csv(input_path, encoding=enc, low_memory=False, keep_default_na=False)
				used_encoding = enc
				break
			except (UnicodeDecodeError, pd.errors.ParserError, Exception) as e:
				last_exc = e
		if df is None:
			# Fallback: read file with errors='replace' and parse from memory
			try:
				import io
				for enc in encodings:
					try:
						with open(input_path, 'r', encoding=enc, errors='replace') as f:
							text = f.read()
						# preserve literal "N/A" strings (don't treat them as NaN)
						df = pd.read_csv(io.StringIO(text), keep_default_na=False)
						used_encoding = f"{enc} (errors=replace)"
						break
					except Exception:
						continue
			except Exception as e:
				# if fallback also failed, raise the original error for debugging
				raise last_exc or e
		# optional: inform which encoding was used
		if used_encoding:
			print(f"Read CSV using encoding: {used_encoding}")

	if col_name not in df.columns:
		raise KeyError(f"Column '{col_name}' not found in {input_path} (columns: {list(df.columns)})")

	# Create normalized helper column to catch trivial variations (trim, collapse spaces, lowercase)
	# Also replace any occurrence of the garbled sequence ﾃ・EEEEﾃ・EEﾃ・EEEE with a single apostrophe
	garbage_seq = 'ﾃ・EEEEﾃ・EEﾃ・EEEE'
	df['__jd_norm'] = (
		df[col_name]
		.astype(str)  # ensures non-strings are handled
		.str.replace(garbage_seq, "'", regex=False)
		.str.strip()
		.str.replace(r'\s+', ' ', regex=True)
		.str.lower()
	)

	# Identify duplicates based on normalized job description, keep the first occurrence
	dup_mask = df['__jd_norm'].duplicated(keep='first')
	num_duplicates = int(dup_mask.sum())
	num_total = len(df)

	# Drop duplicate rows
	if num_duplicates > 0:
		df_dedup = df.loc[~dup_mask].drop(columns='__jd_norm')
	else:
		df_dedup = df.drop(columns='__jd_norm')

	# Save result (CSV if output ends with .csv, otherwise Excel)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if output_path.suffix.lower() == '.csv':
		df_dedup.to_csv(output_path, index=False)
	else:
		df_dedup.to_excel(output_path, index=False)

	# Summary
	print(f"Input file: {input_path}")
	print(f"Rows before: {num_total}")
	print(f"Duplicate Job Descriptions removed: {num_duplicates}")
	print(f"Rows after: {len(df_dedup)}")
	print(f"Saved deduplicated file to: {output_path}")

def main():
	parser = argparse.ArgumentParser(description="Remove rows with duplicated 'Job Description' values from an Excel/CSV file.")
	# make the input positional optional; fall back to module default when not provided
	parser.add_argument('input', nargs='?', type=Path, default=None, help='Path to input Excel/CSV file (optional)')
	parser.add_argument('-o', '--output', type=Path, help='Path for output file. Defaults to input_dedup.[ext]')
	parser.add_argument('-s', '--sheet', default=None, help='Sheet name or index to process (Excel only)')
	parser.add_argument('-c', '--column', default='Job Description', help="Column name to deduplicate on (default: 'Job Description')")
	args = parser.parse_args()

	# use CLI input if provided, otherwise fall back to module-level default
	input_file = args.input if args.input is not None else input_path

	if not input_file.exists():
		print(f"Input file not found: {input_file}", file=sys.stderr)
		sys.exit(2)

	if args.output:
		out_file = args.output
	else:
		out_file = input_file.with_name(input_file.stem + '_dedup' + input_file.suffix)

	# convert sheet arg to int when appropriate
	sheet = None
	if args.sheet is not None:
		if str(args.sheet).isdigit():
			sheet = int(args.sheet)
		else:
			sheet = args.sheet

	try:
		dedupe_job_descriptions(input_file, out_file, sheet_name=sheet, col_name=args.column)
	except Exception as e:
		print(f"Error: {e}", file=sys.stderr)
		sys.exit(1)

if __name__ == '__main__':
	main()
