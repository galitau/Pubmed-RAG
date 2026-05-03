import unittest
import sys
from pathlib import Path

# Add parent directory to path so we can import pdf_generator
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_generator import create_pdf


class TestPdfGenerator(unittest.TestCase):
    def test_smoke_create_pdf_returns_bytes_with_pdf_header(self):
        """Smoke: Basic PDF generation returns a valid PDF binary."""
        pdf_bytes = create_pdf("Quick smoke test")

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_regression_create_pdf_handles_non_latin_text(self):
        """Regression: Non-latin characters should not crash PDF generation."""
        mixed_text = "Clinical summary: efficacy improved by 12% in cohort A. Emoji test: 😀"
        pdf_bytes = create_pdf(mixed_text)

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 100)

    def test_regression_create_pdf_handles_empty_input(self):
        """Regression: Empty text should still produce a readable PDF file."""
        pdf_bytes = create_pdf("")

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 100)

    def test_regression_large_report_generates_larger_output(self):
        """Regression: Larger content should generally increase PDF payload size."""
        small = create_pdf("short report")
        large_input = "\n".join(["Line of findings and references"] * 200)
        large = create_pdf(large_input)

        self.assertGreater(len(large), len(small))


if __name__ == "__main__":
    unittest.main()