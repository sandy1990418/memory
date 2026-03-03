"""Tests for core/llm_client.py — LLM client factory and adapters."""

import os
import unittest
from unittest.mock import MagicMock, patch

from openclaw_memory.core.llm_client import (
    _resolve_key,
    create_llm_client,
)


class TestResolveKey(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_openai_key(self):
        self.assertEqual(_resolve_key("openai"), "sk-test")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-test"})
    def test_anthropic_key(self):
        self.assertEqual(_resolve_key("anthropic"), "ant-test")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gem-test"})
    def test_gemini_key(self):
        self.assertEqual(_resolve_key("gemini"), "gem-test")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_key(self):
        self.assertIsNone(_resolve_key("openai"))

    def test_unknown_provider(self):
        self.assertIsNone(_resolve_key("unknown"))


class TestCreateLLMClientExplicit(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_explicit_provider_missing_key_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            create_llm_client(provider="openai")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_explicit_anthropic_missing_key_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            create_llm_client(provider="anthropic")
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_explicit_gemini_missing_key_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            create_llm_client(provider="gemini")
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm_client(provider="unknown")
        self.assertIn("unknown", str(ctx.exception))


class TestCreateLLMClientAuto(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_auto_no_keys_returns_none(self):
        result = create_llm_client(provider="auto")
        self.assertIsNone(result)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch("openclaw_memory.core.llm_client.OpenAILLMClient")
    def test_auto_detects_openai(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = create_llm_client(provider="auto")
        self.assertIsNotNone(result)
        mock_cls.assert_called_once_with(model="gpt-4o-mini", api_key="sk-test")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-test"}, clear=True)
    @patch("openclaw_memory.core.llm_client.AnthropicLLMClient")
    def test_auto_detects_anthropic(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = create_llm_client(provider="auto")
        self.assertIsNotNone(result)
        mock_cls.assert_called_once_with(model="claude-sonnet-4-20250514", api_key="ant-test")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch("openclaw_memory.core.llm_client.OpenAILLMClient")
    def test_auto_with_custom_model(self, mock_cls):
        mock_cls.return_value = MagicMock()
        create_llm_client(provider="auto", model="gpt-4o")
        mock_cls.assert_called_once_with(model="gpt-4o", api_key="sk-test")


class TestOpenAIAdapter(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch("openclaw_memory.core.llm_client.OpenAILLMClient")
    def test_explicit_openai_creation(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        result = create_llm_client(provider="openai", model="gpt-4o")
        self.assertIs(result, mock_instance)
        mock_cls.assert_called_once_with(model="gpt-4o", api_key="sk-test")


class TestAnthropicAdapter(unittest.TestCase):
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-test"}, clear=True)
    @patch("openclaw_memory.core.llm_client.AnthropicLLMClient")
    def test_explicit_anthropic_creation(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        result = create_llm_client(provider="anthropic", model="claude-sonnet-4-20250514")
        self.assertIs(result, mock_instance)
        mock_cls.assert_called_once_with(model="claude-sonnet-4-20250514", api_key="ant-test")


class TestGeminiAdapter(unittest.TestCase):
    @patch.dict(os.environ, {"GEMINI_API_KEY": "gem-test"}, clear=True)
    @patch("openclaw_memory.core.llm_client.GeminiLLMClient")
    def test_explicit_gemini_creation(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        result = create_llm_client(provider="gemini", model="gemini-2.0-flash")
        self.assertIs(result, mock_instance)
        mock_cls.assert_called_once_with(model="gemini-2.0-flash", api_key="gem-test")


if __name__ == "__main__":
    unittest.main()
