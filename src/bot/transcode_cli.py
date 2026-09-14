import asyncio
import json
import os
import re
import shutil
from pathlib import Path

import click
from dotenv import load_dotenv
import torf

from bot.netcode import BTCategory, QBittorrentClient
from bot.upload_cli import _announce_url, _sanitize_name

load_dotenv()

_TRANSCODE_ROOT = Path("~/Data/transcodes").expanduser()
_AUDIO_EXTS = frozenset({".flac"})
_HIRES_THRESHOLD = 16  # bits strictly greater than this → hi-res

# Strips bracketed format/quality tags so we get a clean base name
_FMT_RE = re.compile(
    r"\s*[\[\({]"
    r"(?:FLAC|MP3|24[\s\-]?bit|16[\s\-]?bit"
    r"|\d{2,3}\s?[kK][hH][zZ]|\d{2,3}\s?[kK](?!B)"
    r"|CBR|VBR|V0|320|[Ll]ossless|WEB(?:-DL)?|CD|Vinyl|SACD|DSD"
    r"|(?:24|16|32)/\d{4,6})"
    r"[^\]\)]*[\]\)}]",
    re.IGNORECASE,
)

# Format tag → (output extension, ffmpeg audio args)
_FORMATS: dict[str, tuple[str, list[str]]] = {
    "FLAC": (
        ".flac",
        ["-c:a", "flac", "-sample_fmt", "s16", "-compression_level", "8"],
    ),
    "320": (
        ".mp3",
        ["-c:a", "libmp3lame", "-b:a", "320k", "-id3v2_version", "3", "-write_id3v1", "1"],
    ),
    "V0": (
        ".mp3",
        ["-c:a", "libmp3lame", "-q:a", "0", "-id3v2_version", "3", "-write_id3v1", "1"],
    ),
}


def _strip_format_tags(name: str) -> str:
    return _FMT_RE.sub("", name).strip()


async def _probe_bits(path: Path) -> int:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    for stream in json.loads(out).get("streams", []):
        if stream.get("codec_type") == "audio":
            bps = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or 16
            return int(bps)
    return 16


async def _run_ffmpeg(
    semaphore: asyncio.Semaphore, src: Path, dst: Path, args: list[str]
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    async with semaphore:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", str(src), *args, "-y", str(dst),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {src.name}")


def _torrent_progress(torrent, filepath, pieces_done, pieces_total):
    pct = int(pieces_done / pieces_total * 100) if pieces_total else 0
    click.echo(f"\r    Hashing: {pct:3d}%", nl=False)
    if pieces_done == pieces_total:
        click.echo()


@click.command()
@click.argument("album_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--workers", "-j", default=None, type=int,
              help="Concurrent ffmpeg processes (default: cpu count)")
@click.option("--no-torrent", is_flag=True, help="Skip torrent creation and seeding")
@click.option("--no-seed", is_flag=True, help="Create torrents but skip adding to qBittorrent")
def transcode_album(album_dir: Path, workers: int | None, no_torrent: bool, no_seed: bool):
    """Transcode a lossless album to all OPS-accepted formats for upload.

    \b
    Always produces:
      [320]   MP3 CBR 320 kbps
      [V0]    MP3 VBR V0
    If source is > 16-bit:
      [FLAC]  FLAC dithered to 16-bit

    Transcodes and torrents are written to ~/Data/transcodes/.
    Torrents are automatically added to qBittorrent for seeding (use --no-seed to skip).
    """
    asyncio.run(_main(album_dir, workers or os.cpu_count() or 4, no_torrent, no_seed))


async def _main(album_dir: Path, workers: int, no_torrent: bool, no_seed: bool) -> None:
    announce_url = _announce_url()

    flac_files = sorted(album_dir.rglob("*.flac"))
    if not flac_files:
        raise click.UsageError(f"No FLAC files found in {album_dir}")

    bits = await _probe_bits(flac_files[0])
    is_hires = bits > _HIRES_THRESHOLD
    click.echo(f"Source: {bits}-bit FLAC — {len(flac_files)} tracks")

    active = {tag: v for tag, v in _FORMATS.items() if tag != "FLAC" or is_hires}
    click.echo(f"Producing: {', '.join(f'[{t}]' for t in active)}")

    base = _sanitize_name(_strip_format_tags(album_dir.name))
    _TRANSCODE_ROOT.mkdir(parents=True, exist_ok=True)

    out_dirs = {tag: _TRANSCODE_ROOT / f"{base} [{tag}]" for tag in active}
    for d in out_dirs.values():
        d.mkdir(exist_ok=True)

    # Copy non-audio files (cover art, logs, cue sheets) to every output dir
    for src in album_dir.rglob("*"):
        if src.is_file() and src.suffix.lower() not in _AUDIO_EXTS:
            rel = src.relative_to(album_dir)
            for out_dir in out_dirs.values():
                dst = out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # Build transcode jobs
    semaphore = asyncio.Semaphore(workers)
    jobs = []
    for tag, (ext, ffargs) in active.items():
        out_dir = out_dirs[tag]
        for src in flac_files:
            dst = out_dir / src.relative_to(album_dir).with_suffix(ext)
            jobs.append(_run_ffmpeg(semaphore, src, dst, ffargs))

    total = len(jobs)
    done = 0
    click.echo(f"Transcoding {total} files across {len(active)} formats ({workers} workers)...")

    async def track(coro):
        nonlocal done
        try:
            await coro
        except RuntimeError as e:
            click.echo(f"\n  Error: {e}", err=True)
            raise
        done += 1
        click.echo(f"\r  {done}/{total}", nl=False)

    await asyncio.gather(*[track(j) for j in jobs])
    click.echo()

    if no_torrent:
        click.echo(f"Transcodes written to {_TRANSCODE_ROOT}")
        return

    click.echo("Creating torrents...")
    torrent_paths: list[tuple[str, Path]] = []
    for tag, out_dir in out_dirs.items():
        torrent_path = _TRANSCODE_ROOT / f"{out_dir.name}.torrent"
        t = torf.Torrent(
            path=out_dir,
            name=out_dir.name,
            trackers=[[announce_url]],
            private=True,
            source="OPS",
        )
        click.echo(f"  [{tag}] hashing...", nl=False)
        t.generate(callback=_torrent_progress, interval=1)
        t.write(torrent_path, overwrite=True)
        click.echo(f"  [{tag}] → {torrent_path.name}")
        torrent_paths.append((tag, torrent_path))

    click.echo(f"\nUpload-ready torrents in: {_TRANSCODE_ROOT}")

    if no_seed:
        return

    seed_path = os.getenv("TRANSCODE_SEED_PATH")
    if not seed_path:
        raise click.UsageError("TRANSCODE_SEED_PATH is not set — needed so qBittorrent can locate the files")

    click.echo("Adding to qBittorrent for seeding...")
    async with QBittorrentClient() as client:
        for tag, torrent_path in torrent_paths:
            memory_code = await client.add_torrent_file(
                torrent_path, BTCategory.Music, save_path=seed_path
            )
            if memory_code:
                click.echo(f"  [{tag}] seeding (tag: sprintboy_{memory_code})")
