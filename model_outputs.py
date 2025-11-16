"""
Count word frequencies in specific and nonspecific model output files.
No seed keywords — pure counts. Prints top 50 for each.
"""

import re
from collections import Counter
from pathlib import Path

SPECIFIC_DATA_PATH = Path("data/model_outputs/specific_data.txt")
NON_SPECIFIC_DATA_PATH = Path("data/model_outputs/nonspecific_data.txt")

# Optional: *light* stopwords
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "on", "in",
    "by", "from", "that", "this", "these", "those", "it", "its",
    "is", "are", "was", "were", "be", "been", "as", "at", "with",
    "we", "our", "you", "they", "their"
}


def load_documents(path: Path) -> list[str]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # strip enclosing quotes
            if len(line) >= 2 and line[0] == line[-1] == '"':
                line = line[1:-1]
            docs.append(line)
    return docs


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z][a-z\-]*", text)
    return [t for t in tokens if t not in STOPWORDS]


def count_words(path: Path, top_n: int = 50):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    docs = load_documents(path)

    counter = Counter()
    for doc in docs:
        counter.update(tokenize(doc))

    print(f"\n=== TOP {top_n} WORDS — {path} ===")
    for word, freq in counter.most_common(top_n):
        print(f"{word:20s} {freq}")

    return counter


def main():
    # Count specific
    specific_counts = count_words(SPECIFIC_DATA_PATH, top_n=50)

    # Count non-specific
    nonspecific_counts = count_words(NON_SPECIFIC_DATA_PATH, top_n=50)


if __name__ == "__main__":
    main()
