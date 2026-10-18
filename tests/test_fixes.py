import os
import unittest
from unittest import mock

from src import indexer, llm
from src.utils import split_chunks_by_tokens


class DummyResponse:
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class LLMModelFallbackTests(unittest.TestCase):
    def test_uses_default_model_when_only_key_is_set(self):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return DummyResponse()

        env = {"OPENROUTER_API_KEY": "test-key"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("src.llm._get_settings", return_value=False),
            mock.patch("src.llm.requests.post", side_effect=fake_post),
        ):
            llm.generate_answer("q", [])

        self.assertEqual(captured["json"]["model"], llm.DEFAULT_MODEL)


class IndexerConnectionTests(unittest.TestCase):
    def test_aborts_without_touching_collection_when_mongo_unreachable(self):
        client = mock.MagicMock()
        client.admin.command.side_effect = Exception("SSL handshake failed")
        with mock.patch("src.indexer.MongoClient", return_value=client):
            with self.assertRaises(RuntimeError) as ctx:
                indexer.index_codebase(
                    repo_path="/nonexistent",
                    db_uri="mongodb+srv://user:secret@host/",
                    db_name="db",
                    collection_name="coll",
                    embed_model_name="unused",
                    chunk_token_target=500,
                    chunk_overlap_tokens=50,
                    subdirs=[],
                )
        self.assertNotIn("secret", str(ctx.exception))
        client.__getitem__.assert_not_called()


class SplitChunksTests(unittest.TestCase):
    def test_chunks_cover_all_lines_in_order(self):
        text = "\n".join(f"line_{i} = {i}" for i in range(200))
        chunks = split_chunks_by_tokens(text, start_line=10, target_tokens=50, overlap_tokens=10)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][1], 10)
        self.assertEqual(chunks[-1][2], 10 + 199)
        for (_, start, end), (_, next_start, _) in zip(chunks, chunks[1:]):
            self.assertLessEqual(start, end)
            self.assertLessEqual(next_start, end + 1)  # contiguous or overlapping
            self.assertGreater(next_start, start)  # always makes progress


if __name__ == "__main__":
    unittest.main()
