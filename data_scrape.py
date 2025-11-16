import os
import io
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import pandas as pd
import funcs

# Config

LINKS_CSV = "data/report_links_2024.csv"
OUT_DIR = "data/processed_txt_2024"
os.makedirs(OUT_DIR, exist_ok=True)

# Text extractors (no local PDF saving)


def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF in memory using PyMuPDF (fitz),
    which is generally more robust than pdfplumber/pdfminer for complex reports.
    """
    text_parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            # "text" gives plain text; you could use "blocks" or "dict" if you want structure
            text_parts.append(page.get_text("text") or "")
    return "\n".join(text_parts)


def extract_html_text_from_bytes(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "html.parser")
    # Remove obvious junk
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return text


def guess_format(url: str, fmt: str | None) -> str:
    """
    Use CSV 'format' as a hint, then URL pattern.
    """
    if isinstance(fmt, str) and fmt.strip():
        return fmt.strip().lower()
    url_lower = url.lower()
    if url_lower.endswith(".pdf"):
        return "pdf"
    if any(ext in url_lower for ext in [".html", ".htm", "sustainability", "esg"]):
        # crude guess, but fine as fallback
        return "html"
    return "pdf"  # default to pdf if unsure


def safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("_", "-"))


# ---------- Load report links ----------

links = pd.read_csv(LINKS_CSV)
# expected columns: ticker, company_name, year, report_url, format (optional)

print("Total rows in links file:", len(links))

# ---------- Main loop: fetch → extract → chunk → write txt ----------

# Browser-like headers to reduce 403s
BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

for idx, row in links.iterrows():
    ticker = str(row["ticker"]).strip()
    year = int(row.get("year", 2024))
    url = row["report_url"]

    if pd.isna(url) or not str(url).strip():
        print(f"[{idx}] {ticker}: missing URL, skipping")
        continue

    fmt = guess_format(url, row.get("format"))

    out_path = os.path.join(OUT_DIR, f"{safe_filename(ticker)}_{year}_chunks.txt")
    if os.path.exists(out_path):
        print(f"[{idx}] {ticker}: already processed -> {out_path}")
        continue

    print(f"[{idx}] Fetching {ticker} {year} ({fmt}) from {url}")

    try:
        resp = requests.get(url, headers=BASE_HEADERS, timeout=45)
        resp.raise_for_status()
    except Exception as e:
        print(f"   !! Failed to download {ticker}: {e}")
        continue

    try:
        if fmt == "pdf":
            raw_text = extract_pdf_text_from_bytes(resp.content)
        elif fmt == "html":
            raw_text = extract_html_text_from_bytes(resp.content)
        else:
            # fallback guess
            if url.lower().endswith(".pdf"):
                raw_text = extract_pdf_text_from_bytes(resp.content)
            else:
                raw_text = extract_html_text_from_bytes(resp.content)
    except Exception as e:
        print(f"   !! Failed to extract text for {ticker}: {e}")
        continue

    if not raw_text or not raw_text.strip():
        print(f"   !! Empty text for {ticker}, skipping write")
        continue

    # Chunk into your training format
    chunks = funcs.split_and_quote(raw_text, width=500)

    if not chunks:
        print(f"   !! No chunks produced for {ticker}, skipping write")
        continue

    # Write per-company TXT with quoted lines
    with open(out_path, "w", encoding="utf-8") as f:
        for line in chunks:
            f.write(line + "\n")

    print(f"   -> Wrote {len(chunks)} chunks to {out_path}")
