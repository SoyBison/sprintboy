import asyncio
import random
import re

import aiohttp
import click
from thefuzz import fuzz

from bot.netcode import BTCategory, QBittorrentClient
from bot.transcode_cli import _strip_format_tags

_FLAC_RE = re.compile(r"\bFLAC\b", re.I)
_HIRES_RE = re.compile(r"24[\s\-]?bit|24/\d{2,6}|24[\s\-]?\d{2,3}\s?[kK](?!B)", re.I)

_FORMAT_RES: list[tuple[str, re.Pattern]] = [
    ("V0",   re.compile(r"\bV0\b",  re.I)),
    ("V2",   re.compile(r"\bV2\b",  re.I)),
    ("320",  re.compile(r"\b320\b", re.I)),
    ("FLAC", re.compile(r"\bFLAC\b", re.I)),
]

_PROGRESS_WIDTH = 60


def _formats_in(filename: str) -> set[str]:
    return {tag for tag, pat in _FORMAT_RES if pat.search(filename)}


def _print_above_progress(lines: list[str]) -> None:
    """Clear the progress line, print result lines, then leave cursor ready for progress."""
    click.echo(f"\r{' ' * _PROGRESS_WIDTH}\r", nl=False)
    for line in lines:
        click.echo(line)


@click.command()
@click.option("--concurrency", "-c", default=3, show_default=True,
              help="Concurrent workers — 409s retry with exponential backoff")
@click.option("--threshold", "-t", default=75, show_default=True,
              help="Fuzzy name-match score 0–100 to consider a result the same release")
def find_opportunities(concurrency: int, threshold: int):
    """Find FLAC torrents in qBittorrent that have missing transcodes on the tracker.

    \b
    For each completed FLAC in your Music library:
      • Searches Jackett for existing uploads of that release
      • Reports formats that are absent: [320], [V0], and [FLAC] for 24-bit sources

    Opportunities are printed as they are found, above the progress counter.
    Workers that receive a 409 (search slot busy) back off with exponential
    jitter and retry automatically.
    """
    asyncio.run(_main(concurrency, threshold))


async def _main(concurrency: int, threshold: int) -> None:
    async with QBittorrentClient() as client:
        all_torrents = await client.list_torrents(BTCategory.Music)
        flac_torrents = [t for t in all_torrents if _FLAC_RE.search(t.name)]

        if not flac_torrents:
            click.echo("No completed FLAC torrents found in Music library.")
            return

        total = len(flac_torrents)
        click.echo(f"Found {total} FLAC torrents. Searching Jackett...")

        queue: asyncio.Queue = asyncio.Queue()
        for torrent in flac_torrents:
            await queue.put(torrent)

        done = 0

        async def worker() -> None:
            nonlocal done
            while True:
                torrent = await queue.get()
                try:
                    query = _strip_format_tags(torrent.name)
                    hires = bool(_HIRES_RE.search(torrent.name))
                    needed = {"320", "V0"} | ({"FLAC"} if hires else set())

                    backoff = 1.0
                    while True:
                        try:
                            results = await client.search(query, BTCategory.Music)
                            break
                        except aiohttp.ClientResponseError as e:
                            if e.status != 409:
                                raise
                            await asyncio.sleep(backoff + random.uniform(0, 1))
                            backoff = min(backoff * 2, 60)

                    existing: set[str] = set()
                    for r in results.results:
                        if fuzz.partial_ratio(query.lower(), r.fileName.lower()) >= threshold:
                            existing |= _formats_in(r.fileName)

                    missing = needed - existing
                    done += 1

                    if missing:
                        tags = "  ".join(f"[{m}]" for m in sorted(missing))
                        note = "  ← 24-bit" if hires else ""
                        _print_above_progress([
                            f"  {query}{note}",
                            f"    needs: {tags}",
                            f"    path:  {torrent.content_path}",
                        ])

                except Exception as e:
                    done += 1
                    _print_above_progress([f"  Warning: {e}"])
                finally:
                    click.echo(f"\r  {done}/{total}", nl=False)
                    queue.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await queue.join()
        for task in tasks:
            task.cancel()
        click.echo()
