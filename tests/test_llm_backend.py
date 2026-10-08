"""
Unit tests for backend selection.

The agent can run on the Anthropic API or on a self-hosted ollama, and picking
the wrong one is the kind of mistake that only shows up as a confusing runtime
error, so the switch and its validation are covered here.
"""

import pytest

from bot.config import Config


@pytest.fixture
def restore_config():
    saved = {
        name: getattr(Config, name)
        for name in (
            "LLM_PROVIDER",
            "OLLAMA_API_URL",
            "OLLAMA_MODEL",
            "ANTHROPIC_API_KEY",
        )
    }
    yield
    for name, value in saved.items():
        setattr(Config, name, value)


def test_ollama_needs_a_url(restore_config):
    Config.LLM_PROVIDER = "ollama"
    Config.OLLAMA_API_URL = ""
    with pytest.raises(AssertionError, match="OLLAMA_API_URL"):
        Config.validate_llm()


def test_anthropic_needs_a_key(restore_config):
    Config.LLM_PROVIDER = "anthropic"
    Config.ANTHROPIC_API_KEY = ""
    with pytest.raises(AssertionError, match="ANTHROPIC_API_KEY"):
        Config.validate_llm()


def test_unknown_provider_is_rejected(restore_config):
    Config.LLM_PROVIDER = "llamafile"
    with pytest.raises(AssertionError, match="LLM_PROVIDER"):
        Config.validate_llm()


def test_anthropic_does_not_need_an_ollama_url(restore_config):
    """A missing ollama URL used to fail startup even on the API backend."""
    Config.LLM_PROVIDER = "anthropic"
    Config.ANTHROPIC_API_KEY = "sk-test"
    Config.OLLAMA_API_URL = ""
    Config.validate_llm()


def test_build_chat_model_uses_the_configured_ollama_model(restore_config, monkeypatch):
    from bot.llm import OllamaChat, build_chat_model

    Config.LLM_PROVIDER = "ollama"
    Config.OLLAMA_API_URL = "http://ollama.invalid:11434"
    Config.OLLAMA_MODEL = "gpt-oss:20b"
    monkeypatch.setattr(Config, "OLLAMA_NUM_CTX", 4096)
    llm = build_chat_model()

    assert isinstance(llm, OllamaChat)
    assert llm.model == "gpt-oss:20b"
    assert llm.base_url == "http://ollama.invalid:11434"
    assert llm.num_ctx == 4096
