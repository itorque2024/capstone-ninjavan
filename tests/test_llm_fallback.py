"""Tests for the LLM call helper and Groq fallback logic."""
import pytest
from unittest.mock import patch, MagicMock


class TestLlmFallback:
    def _make_gemini_429(self):
        err = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
        return err

    def test_returns_gemini_response_when_available(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "Hello from Gemini"

        with patch("src.agents.chatbot._llm._client", mock_client):
            from src.agents.chatbot._llm import llm_call
            result = llm_call("test prompt")
        assert result == "Hello from Gemini"

    def test_falls_back_to_groq_on_429(self):
        mock_gemini = MagicMock()
        mock_gemini.models.generate_content.side_effect = self._make_gemini_429()

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value.choices[0].message.content = "Hello from Groq"

        with patch("src.agents.chatbot._llm._client", mock_gemini), \
             patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}), \
             patch("groq.Groq", return_value=mock_groq_client):
            from importlib import reload
            import src.agents.chatbot._llm as llm_mod
            result = llm_mod.llm_call("test prompt")

        assert result == "Hello from Groq"

    def test_raises_if_no_groq_key_on_429(self):
        mock_gemini = MagicMock()
        mock_gemini.models.generate_content.side_effect = self._make_gemini_429()

        with patch("src.agents.chatbot._llm._client", mock_gemini), \
             patch.dict("os.environ", {}, clear=True):
            import src.agents.chatbot._llm as llm_mod
            with pytest.raises(Exception, match="429"):
                llm_mod.llm_call("test prompt")

    def test_non_429_error_propagates_immediately(self):
        mock_gemini = MagicMock()
        mock_gemini.models.generate_content.side_effect = Exception("503 UNAVAILABLE")

        with patch("src.agents.chatbot._llm._client", mock_gemini):
            import src.agents.chatbot._llm as llm_mod
            with pytest.raises(Exception, match="503"):
                llm_mod.llm_call("test prompt")


class TestLlmCallInterface:
    def test_max_tokens_parameter_accepted(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "ok"
        with patch("src.agents.chatbot._llm._client", mock_client):
            from src.agents.chatbot._llm import llm_call
            result = llm_call("prompt", max_tokens=256)
        assert result == "ok"
