import os
import json
import unittest
from unittest import mock

from src import llm
from src.prompts import load_system_prompt


class DummyResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class LLMTests(unittest.TestCase):
    def test_generate_answer_builds_messages(self):
        old_key = os.environ.get("OPENROUTER_API_KEY")
        old_model = os.environ.get("OPENROUTER_MODEL")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        os.environ["OPENROUTER_MODEL"] = "test-model"

        code_chunks = [
            {
                "text": "def hello():\n    return 'hi'\n",
                "meta": {"path": "foo.py", "start_line": 1, "end_line": 2},
            }
        ]

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return DummyResponse(200, {"choices": [{"message": {"content": "ok"}}]})

        try:
            with mock.patch("src.llm.requests.post", side_effect=fake_post):
                answer = llm.generate_answer("What does hello do?", code_chunks)
        finally:
            if old_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_key
            if old_model is None:
                os.environ.pop("OPENROUTER_MODEL", None)
            else:
                os.environ["OPENROUTER_MODEL"] = old_model

        self.assertEqual(answer, "ok")
        self.assertEqual(captured["json"]["model"], "test-model")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")

        messages = captured["json"]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], load_system_prompt())
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("File: foo.py (L1-L2):", messages[1]["content"])
        self.assertIn("def hello()", messages[1]["content"])
