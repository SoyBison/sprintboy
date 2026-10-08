import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot import tags
from bot.config import Config
from bot.netcode import LastFMClient, LastFMError


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DECISION_LOG_PATH", str(tmp_path / "decisions.jsonl"))
    return tmp_path


def _fake(tag_list=None, error=None):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get_top_tags = AsyncMock(return_value=tag_list, side_effect=error)
    return client


TAGS = [
    ("rock", 900), ("seen live", 800), ("Shoegaze", 700), ("female vocalists", 650),
    ("60s", 600), ("japanese", 500), ("japanese city pop", 400), ("favorites", 300),
    ("rock", 200), ("chill", 100),
]


@pytest.mark.asyncio
async def test_cache_written_and_reused(cache_dir):
    client = _fake(TAGS)
    with patch("bot.tags.LastFMClient", return_value=client):
        first = await tags.tag_vocabulary()
        second = await tags.tag_vocabulary()
    assert first == second
    assert client.get_top_tags.await_count == 1
    saved = json.loads((cache_dir / "lastfm_tags.json").read_text())
    assert saved["tags"] == first
    assert "fetched" in saved


@pytest.mark.asyncio
async def test_noise_filtered_decades_kept():
    with patch("bot.tags.LastFMClient", return_value=_fake(TAGS)):
        vocab = await tags.tag_vocabulary()
    assert vocab == ["rock", "shoegaze", "60s", "japanese city pop", "chill"]


@pytest.mark.asyncio
async def test_stale_cache_is_refetched(cache_dir):
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    (cache_dir / "lastfm_tags.json").write_text(json.dumps({"fetched": old, "tags": ["old"]}))
    client = _fake([("new", 1)])
    with patch("bot.tags.LastFMClient", return_value=client):
        assert await tags.tag_vocabulary() == ["new"]
    assert client.get_top_tags.await_count == 1


@pytest.mark.asyncio
async def test_stale_cache_used_when_lastfm_fails(cache_dir):
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    (cache_dir / "lastfm_tags.json").write_text(json.dumps({"fetched": old, "tags": ["old"]}))
    with patch("bot.tags.LastFMClient", return_value=_fake(error=LastFMError("down"))):
        assert await tags.tag_vocabulary() == ["old"]


@pytest.mark.asyncio
async def test_fallback_without_cache(cache_dir):
    with patch("bot.tags.LastFMClient", return_value=_fake(error=LastFMError("down"))):
        vocab = await tags.tag_vocabulary()
    assert vocab == tags.FALLBACK_TAGS
    assert "shoegaze" in vocab and "chill" in vocab
    assert not (cache_dir / "lastfm_tags.json").exists()


@pytest.mark.asyncio
async def test_top_tags_parsed():
    client = LastFMClient()
    body = {"tags": {"tag": [{"name": "rock", "reach": "5"}, {"name": "pop", "taggings": "9"}, {"name": ""}]}}
    with patch.object(client, "_get", AsyncMock(return_value=body)) as get:
        assert await client.get_top_tags(10) == [("pop", 9), ("rock", 5)]
    get.assert_awaited_once_with("chart.getTopTags", limit=10)


VOCAB = ["rock", "indie", "jazz", "chillout", "chill", "shoegaze", "dream pop", "metal", "new wave"]


def test_named_tag_comes_first():
    out = tags.vocabulary_matches("throw some new shoegaze at me", VOCAB)
    assert out[0] == "shoegaze"
    assert set(out) == set(VOCAB)


def test_mood_offers_chill_tags():
    out = tags.vocabulary_matches("something chill to work to", VOCAB, limit=3)
    assert out[0] == "chill"
    assert "chillout" in out


def test_multi_word_tag_and_limit():
    out = tags.vocabulary_matches("some dream pop please", VOCAB, limit=2)
    assert out[0] == "dream pop"
    assert len(out) == 2


def test_popular_fill_in_order():
    assert tags.vocabulary_matches("hello", VOCAB, limit=3) == ["rock", "indie", "jazz"]
