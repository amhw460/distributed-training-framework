# run with: python -m unittest tests.test_cleaning

import unittest

from framework.data.cleaning import DocumentCleaner, clean_document, normalize_text


class CleaningTests(unittest.TestCase):
    def test_normalize_whitespace_and_paragraphs(self):
        text = "  First\t  sentence. \r\n \r\n\r\n   Second   sentence.  \r"
        self.assertEqual(normalize_text(text), "First sentence.\n\nSecond sentence.")

    def test_empty_and_punctuation_only(self):
        self.assertEqual(clean_document(" \t\n", min_words=1), (None, "empty"))
        self.assertEqual(clean_document(" !! ?? ", min_words=0), (None, "empty"))

    def test_short_document(self):
        self.assertEqual(clean_document("two words", min_words=3), (None, "too_short"))
        self.assertEqual(clean_document("two words", min_words=2), ("two words", None))

    def test_invalid_input_and_threshold(self):
        with self.assertRaises(TypeError):
            normalize_text(None)
        with self.assertRaises(ValueError):
            clean_document("hello", min_words=-1)
        with self.assertRaises(ValueError):
            DocumentCleaner(min_words=1.5)

    def test_deduplicates_normalized_text(self):
        cleaner = DocumentCleaner(min_words=3)
        self.assertEqual(cleaner.clean("one  two\nthree"), ("one two\nthree", None))
        self.assertEqual(cleaner.clean(" one two\nthree "), (None, "duplicate"))
        self.assertEqual(cleaner.clean("another four words"), ("another four words", None))

    def test_rejected_documents_not_marked_seen(self):
        cleaner = DocumentCleaner(min_words=3)
        self.assertEqual(cleaner.clean("short"), (None, "too_short"))
        self.assertEqual(cleaner.clean("short longer enough"), ("short longer enough", None))

    def test_summary_counts(self):
        cleaner = DocumentCleaner(min_words=2)
        for text in ("", "one", "two words", "two  words", "three words"):
            cleaner.clean(text)
        self.assertEqual(cleaner.stats(), {
            "processed": 5,
            "kept": 2,
            "empty": 1,
            "too_short": 1,
            "duplicate": 1,
        })


if __name__ == "__main__":
    unittest.main()
