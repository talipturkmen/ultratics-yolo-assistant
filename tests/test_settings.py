import unittest

from src.settings import Settings


class SettingsTests(unittest.TestCase):
    def _base_kwargs(self):
        return {
            "MONGODB_URI": "mongodb://localhost:27017",
            "MONGODB_DB": "db",
            "MONGODB_COLLECTION": "coll",
            "OPENROUTER_API_KEY": "key",
            "ULTRALYTICS_PATH": "/tmp/ultralytics",
        }

    def test_index_subdirs_list_comma(self):
        settings = Settings(**{**self._base_kwargs(), "INDEX_SUBDIRS": "a,b, c "})
        self.assertEqual(settings.index_subdirs_list(), ["a", "b", "c"])

    def test_index_subdirs_list_json(self):
        settings = Settings(**{**self._base_kwargs(), "INDEX_SUBDIRS": '["a","b"]'})
        self.assertEqual(settings.index_subdirs_list(), ["a", "b"])

    def test_index_subdirs_list_empty(self):
        settings = Settings(**{**self._base_kwargs(), "INDEX_SUBDIRS": ""})
        self.assertEqual(settings.index_subdirs_list(), [])
