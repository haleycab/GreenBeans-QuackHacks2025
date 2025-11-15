import pandas as pd
import textwrap


def split_and_quote(text, width=500):
    """
    Split text into 500-character chunks.
    Each line:
      - starts with exactly one "
      - ends with exactly one "
      - contains NO internal "
    """
    if pd.isna(text):
        return []

    # Clean out newlines + internal quotes
    text = (
        text.replace("\n", " ")
            .replace("\r", " ")
            .replace('"', '')
    )

    # Create 500-character chunks
    chunks = textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        drop_whitespace=True
    )

    # Add exactly one " at start and end
    return [f"\"{chunk}\"" for chunk in chunks]