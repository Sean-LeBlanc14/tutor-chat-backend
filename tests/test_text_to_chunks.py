import unittest
import sys
import os

# Allow imports from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from texts_to_chunks import split_text  # now works if texts_to_chunks.py is in repo root


class TestSplitText(unittest.TestCase):

    def test_regular_chunking(self):
        text = "A" * 1200
        chunks = split_text(text, chunk_size=500, overlap=100)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "A" * 500)
        self.assertEqual(chunks[1], "A" * 500)
        self.assertEqual(chunks[2], "A" * 400)

    def test_overlap_effectiveness(self):
        text = "1234567890" * 100  # 1000 characters
        chunks = split_text(text, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][-50:], chunks[1][:50])

    def test_short_text(self):
        text = "short text"
        chunks = split_text(text, chunk_size=500, overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "short text")

    def test_empty_text(self):
        text = ""
        chunks = split_text(text, chunk_size=500, overlap=100)
        self.assertEqual(chunks, [])

    def test_whitespace_only(self):
        text = "   \n  \t   "
        chunks = split_text(text, chunk_size=500, overlap=100)
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
