import unittest

from src.prompts import SYSTEM_PROMPT, load_system_prompt


class PromptsTests(unittest.TestCase):
    def test_system_prompt_loads(self):
        prompt = load_system_prompt()
        self.assertTrue(prompt)
        self.assertIn("Ultralytics YOLO", prompt)

    def test_system_prompt_constant_matches_loader(self):
        self.assertEqual(SYSTEM_PROMPT, load_system_prompt())
