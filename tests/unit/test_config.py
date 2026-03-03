"""Tests for the new Pydantic-based config module."""

import os
import unittest


class TestAppSettings(unittest.TestCase):
    def test_default_settings(self):
        from openclaw_memory.config import AppSettings

        s = AppSettings()
        self.assertEqual(s.pg_dsn, "")
        self.assertEqual(s.embedding_provider, "auto")
        self.assertEqual(s.sensory_pre_compress, True)
        self.assertEqual(s.sensory_topic_segment, True)
        self.assertEqual(s.search_vector_weight, 0.7)
        self.assertEqual(s.search_text_weight, 0.3)
        self.assertTrue(s.search_temporal_decay_enabled)
        self.assertTrue(s.search_mmr_enabled)
        self.assertEqual(s.search_mmr_lambda, 0.7)
        self.assertEqual(s.working_memory_max_messages, 20)
        self.assertEqual(s.host, "0.0.0.0")
        self.assertEqual(s.port, 8000)

    def test_env_override(self):
        os.environ["OPENCLAW_PG_DSN"] = "postgresql://test:test@localhost/test"
        os.environ["OPENCLAW_PORT"] = "9000"
        try:
            from openclaw_memory.config import AppSettings

            s = AppSettings()
            self.assertEqual(s.pg_dsn, "postgresql://test:test@localhost/test")
            self.assertEqual(s.port, 9000)
        finally:
            del os.environ["OPENCLAW_PG_DSN"]
            del os.environ["OPENCLAW_PORT"]

    def test_get_settings_returns_instance(self):
        from openclaw_memory.config import AppSettings, get_settings

        s = get_settings()
        self.assertIsInstance(s, AppSettings)

    def test_validation_constraints(self):
        from openclaw_memory.config import AppSettings
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            AppSettings(search_vector_weight=2.0)

        with self.assertRaises(ValidationError):
            AppSettings(search_mmr_lambda=-0.5)


if __name__ == "__main__":
    unittest.main()
