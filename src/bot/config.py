import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "")
    # Add more config as needed
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "")

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
