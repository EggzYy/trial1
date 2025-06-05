import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trial_integrated4 import format_glossary

class TestExample(unittest.TestCase):

    def test_always_passes(self):
        """A simple placeholder test that always passes."""
        self.assertEqual(True, True)

    def test_format_glossary_empty(self):
        """format_glossary should return an empty string for an empty list."""
        self.assertEqual(format_glossary([]), "")

    def test_format_glossary_single_item(self):
        """format_glossary should correctly format a single glossary item."""
        item = [{"source": "term", "target": "Begriff", "notes": "example"}]
        expected = (
            "<GlossaryItem><Source>term</Source><Target>Begriff</Target>"
            "<Notes>example</Notes></GlossaryItem>"
        )
        self.assertEqual(format_glossary(item), expected)

if __name__ == '__main__':
    unittest.main()
