""" Test to make sure  extracting files and converting them to txt works """
import os
import sys
import tempfile
import unittest
from pathlib import Path
from extract_box_texts import extract_text, process_folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class TestExtractBoxTexts(unittest.TestCase):
    """ Sets the set up and clean up for the test directories for the unit tests """
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "input"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_text_basic_formats(self):
        """ This is a basic test function with different kinds of files """
        files = {
            "sample.txt": "Hello, this is a plain text file.",
            "config.conf": "debug=true",
            "script.py": "print('Hello')",
            "experiment.psyexp": "<Experiment>Stimulus</Experiment>"
        }

        for name, content in files.items():
            file_path = self.input_dir / name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            extracted = extract_text(str(file_path))
            self.assertIn(content, extracted)

    def test_process_folder_creates_unique_txt_files(self):
        """ This is a test to seehow it handles duplicates when it comes to file names """
        # Create two files with the same base name
        file1 = self.input_dir / "duplicate.txt"
        file2 = self.input_dir / "duplicate.conf"
        file1.write_text("Text from file 1", encoding="utf-8")
        file2.write_text("Text from file 2", encoding="utf-8")

        # Run the function
        process_folder(str(self.input_dir), str(self.output_dir))

        # Get all output files
        output_files = list(self.output_dir.glob("*.txt"))
        output_filenames = [f.name for f in output_files]

        self.assertIn("duplicate.txt", output_filenames)
        self.assertIn("duplicate_1.txt", output_filenames)
        self.assertEqual(len(output_filenames), 2)

        contents = [f.read_text() for f in output_files]
        self.assertTrue(any("Text from file 1" in c for c in contents))
        self.assertTrue(any("Text from file 2" in c for c in contents))


if __name__ == '__main__':
    unittest.main()
