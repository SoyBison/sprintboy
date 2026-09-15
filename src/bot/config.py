import logging
import os

from dotenv import load_dotenv

load_dotenv()


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

    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "")

    # Used to ground the agent in real discographies and to canonicalise artist
    # and album names before checking Plex. Optional: without it the Last.fm
    # tools error out and check_for_album falls back to exact-name matching.
    LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")

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
        assert cls.OLLAMA_API_URL, "OLLAMA_API_URL is not set"
        assert cls.DISCORD_GUILD_ID, "DISCORD_GUILD_ID is not set"
        if not cls.LASTFM_API_KEY:
            logging.warning(
                "LASTFM_API_KEY is not set: recommendations will fall back to the "
                "model's own knowledge and album de-duplication will be weaker. "
                "Create a key at https://www.last.fm/api/account/create"
            )
