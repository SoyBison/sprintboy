import functools
import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# These log a line per HTTP request and per trace upload, which buries the
# bot's own output. The deployed container only keeps 50m of logs, so this is
# the difference between being able to read them and not.
_NOISY_LOGGERS = (
    "httpcore",
    "httpx",
    "urllib3",
    "langsmith",
    "discord.gateway",
    "discord.client",
    "discord.http",
)


def setup_logging(level: str | int | None = None) -> None:
    """Configure logging at the requested level, minus the third party noise."""
    logging.basicConfig(level=level or Config.LOG_LEVEL, format=LOG_FORMAT, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


class Config:
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "")
    # Comma separated channel ids the bot will answer in. Empty means every
    # channel it can see.
    DISCORD_CHANNEL_IDS = [
        channel_id.strip()
        for channel_id in os.getenv("DISCORD_CHANNEL_IDS", "").split(",")
        if channel_id.strip()
    ]
    # Add more config as needed
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Which backend the agent runs on: "anthropic" for the API, "ollama" for a
    # self-hosted model.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
    # Ollama defaults to a 4k context, which the system prompt and the nine tool
    # schemas fill before the user has said anything. Too small a window and the
    # model silently forgets the search results it is meant to be choosing from.
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
    # Check at startup that the model has actually been pulled, so a typo fails
    # loudly instead of as a 404 on the first message. Turn off to start the bot
    # while the model server is still coming up.
    OLLAMA_VALIDATE_MODEL = (
        os.getenv("OLLAMA_VALIDATE_MODEL", "true").strip().lower() == "true"
    )

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

    # Used to ground the agent in real discographies and to canonicalise artist
    # and album names before checking Plex. Optional: without it the Last.fm
    # tools error out and check_for_album falls back to exact-name matching.
    LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")

    # Decision models: typed choice/score/yes-no questions answered with
    # probabilities, used for routing and same-release checks. Both backends
    # speak TypeSafe's /v1/systemone format. PRIMARY's answers drive the bot;
    # SHADOW is asked the same thing in the background and only logged, which
    # is what builds the comparison and distillation data. "off" disables one.
    DECISION_PRIMARY = os.getenv("DECISION_PRIMARY", "jev").strip().lower()
    DECISION_SHADOW = os.getenv("DECISION_SHADOW", "laya").strip().lower()
    TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
    TYPESAFE_URL = os.getenv("TYPESAFE_URL", "https://api.typesafe.ai")
    TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")
    OLLAYA_URL = os.getenv("OLLAYA_URL", "")
    OLLAYA_API_KEY = os.getenv("OLLAYA_API_KEY", "")
    OLLAYA_DECISION_MODEL = os.getenv("OLLAYA_DECISION_MODEL", "laya")
    # One JSON line per decision: state, questions, every backend's answers and
    # latency. /app/data is the mounted volume in the container.
    DECISION_LOG_PATH = os.getenv("DECISION_LOG_PATH", "data/decisions.jsonl")

    # The commit the bot is running. The Dockerfile bakes this in at build time
    # because the deployed container holds no .git to ask.
    GIT_SHA = os.getenv("GIT_SHA", "")

    @classmethod
    def validate(cls):
        assert cls.DISCORD_TOKEN, "DISCORD_TOKEN is not set"
        assert cls.ENVIRONMENT in [
            "development",
            "production",
        ], "ENVIRONMENT must be 'development' or 'production'"
        assert cls.LOG_LEVEL in [
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ], "LOG_LEVEL must be 'DEBUG', 'INFO', 'WARNING', or 'ERROR'"
        cls.validate_llm()
        assert cls.DISCORD_GUILD_ID, "DISCORD_GUILD_ID is not set"
        if not cls.LASTFM_API_KEY:
            logging.warning(
                "LASTFM_API_KEY is not set: recommendations will fall back to the "
                "model's own knowledge and album de-duplication will be weaker. "
                "Create a key at https://www.last.fm/api/account/create"
            )

    @classmethod
    def validate_llm(cls):
        """Check the settings the chosen backend needs, for the CLIs too."""
        assert cls.LLM_PROVIDER in [
            "anthropic",
            "ollama",
        ], "LLM_PROVIDER must be 'anthropic' or 'ollama'"
        if cls.LLM_PROVIDER == "ollama":
            assert cls.OLLAMA_API_URL, "OLLAMA_API_URL is not set"
            assert cls.OLLAMA_MODEL, "OLLAMA_MODEL is not set"
        else:
            assert cls.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set"


@functools.cache
def git_sha() -> str:
    """The commit this process is running, or "unknown".

    GIT_SHA wins, since in the container it is the only truth available. Asking
    git is the fallback for running out of a checkout, where the working tree
    can also be ahead of the last commit -- hence the "-dirty" marker, without
    which a local edit looks like whatever was committed before it.
    """
    if Config.GIT_SHA:
        return Config.GIT_SHA
    repo = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        # No git binary, or not a checkout: the container before GIT_SHA existed.
        return "unknown"
    return f"{sha}-dirty" if dirty else sha
