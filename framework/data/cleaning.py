import hashlib
import re
from collections import Counter


# normalize whitespace while keeping paragraph breaks
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Document text must be a string")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# normalize and check the document, return the cleaned text or a rejection reason
def clean_document(text: str, min_words: int = 50) -> tuple[str | None, str | None]:
    if not isinstance(min_words, int) or isinstance(min_words, bool) or min_words < 0:
        raise ValueError("min_words must be a nonnegative integer")

    cleaned = normalize_text(text)
    if not cleaned or not any(char.isalnum() for char in cleaned):
        return None, "empty"
    if len(cleaned.split()) < min_words:
        return None, "too_short"

    return cleaned, None

# keep track of duplicate documents and why we reject them
class DocumentCleaner:

    # initialize the cleaner and its counters
    def __init__(self, min_words: int = 50):
        if not isinstance(min_words, int) or isinstance(min_words, bool) or min_words < 0:
            raise ValueError("min_words must be a nonnegative integer")

        self.min_words = min_words
        self.seen_hashes = set()
        self.counts = Counter()

    # clean one document and check if we have already seen it
    def clean(self, text: str) -> tuple[str | None, str | None]:
        cleaned, reason = clean_document(text, self.min_words)
        self.counts["processed"] += 1

        if reason is not None:
            self.counts[reason] += 1
            return None, reason

        # hash normalized text, so differences in whitespace don't bypass dedup
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if digest in self.seen_hashes:
            self.counts["duplicate"] += 1
            return None, "duplicate"

        self.seen_hashes.add(digest)
        self.counts["kept"] += 1
        return cleaned, None

    # get the counts for each cleaning result
    def stats(self) -> dict[str, int]:
        return {
            "processed": self.counts["processed"],
            "kept": self.counts["kept"],
            "empty": self.counts["empty"],
            "too_short": self.counts["too_short"],
            "duplicate": self.counts["duplicate"],
        }
