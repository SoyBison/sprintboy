"""
Unit tests for the commit /ping reports.

The point of the command is answering "is what I deployed actually running", so
a stale or silently wrong answer is worse than no answer: GIT_SHA from the image
has to win over anything git says, and a failure has to be visible as "unknown".
"""

import subprocess

import pytest

from bot.config import Config, git_sha


@pytest.fixture(autouse=True)
def restore_sha():
    saved = Config.GIT_SHA
    git_sha.cache_clear()
    yield
    Config.GIT_SHA = saved
    git_sha.cache_clear()


def test_baked_in_sha_wins(monkeypatch):
    """In the container git is neither present nor meaningful."""
    Config.GIT_SHA = "abc1234"

    def explode(*args, **kwargs):
        raise AssertionError("git must not be consulted when GIT_SHA is set")

    monkeypatch.setattr(subprocess, "run", explode)
    assert git_sha() == "abc1234"


def test_checkout_sha_is_marked_when_the_tree_is_dirty(monkeypatch):
    Config.GIT_SHA = ""
    outputs = {("rev-parse", "--short", "HEAD"): "500e55e", ("status", "--porcelain"): " M src/bot/main.py"}

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[tuple(cmd[1:])], stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_sha() == "500e55e-dirty"


def test_clean_checkout_is_the_bare_sha(monkeypatch):
    Config.GIT_SHA = ""
    outputs = {("rev-parse", "--short", "HEAD"): "500e55e", ("status", "--porcelain"): ""}

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[tuple(cmd[1:])], stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_sha() == "500e55e"


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("no git"), subprocess.CalledProcessError(128, "git")],
)
def test_no_git_and_no_env_is_unknown(monkeypatch, failure):
    """An image built before GIT_SHA existed, or a source copy with no .git."""
    Config.GIT_SHA = ""

    def fake_run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_sha() == "unknown"
