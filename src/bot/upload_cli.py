import asyncio
import os
import re
import shutil
import unicodedata
from pathlib import Path

import click
from dotenv import load_dotenv
import torf

from bot.netcode import BTCategory, QBittorrentClient

load_dotenv()


def _sanitize_name(name: str) -> str:
    """NFC-normalize and strip characters that are illegal on NTFS/Samba shares."""
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r'[\\/:*?"<>|]', "-", name)
    name = re.sub(r"-{2,}", "-", name)
    return name.strip(". ")


def _announce_url() -> str:
    url = os.getenv("ANNOUNCE_URL")
    if not url:
        raise click.UsageError("ANNOUNCE_URL is not set in config")
    return url


def _nfs_share_path() -> Path:
    path = os.getenv("NFS_SHARE_PATH")
    if not path:
        raise click.UsageError("NFS_SHARE_PATH is not set in config")
    return Path(path)


@click.command()
@click.argument("album_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write the .torrent file (default: next to album_dir)",
)
@click.option("--dry-run", is_flag=True, help="Skip NFS copy and qBittorrent submission")
@click.option("--no-seed", is_flag=True, help="Skip adding the torrent to qBittorrent")
def upload_album(album_dir: Path, output: Path | None, dry_run: bool, no_seed: bool):
    """Create a torrent for ALBUM_DIR, copy it to the NFS share, and seed via qBittorrent."""
    announce_url = _announce_url()
    nfs_path = _nfs_share_path()

    album_name = _sanitize_name(album_dir.name)
    torrent_path = output or (album_dir.parent / f"{album_name}.torrent")
    nfs_dest = nfs_path / album_name

    click.echo(f"Building torrent for: {album_dir}")
    t = torf.Torrent(path=album_dir, name=album_name, trackers=[[announce_url]], private=True, source="OPS")
    t.generate(callback=_progress_cb, interval=1)
    t.write(torrent_path, overwrite=True)
    click.echo(f"Torrent written to: {torrent_path}")

    if dry_run:
        click.echo("Dry run — skipping NFS copy and qBittorrent submission.")
        return

    if album_dir.resolve() == nfs_dest.resolve():
        click.echo("Album is already on the NFS share, skipping copy.")
    else:
        click.echo(f"Copying album to NFS share: {nfs_dest}")
        shutil.copytree(album_dir, nfs_dest, dirs_exist_ok=True)
        click.echo("Copy complete.")

    if not no_seed:
        asyncio.run(_submit(torrent_path))


async def _submit(torrent_path: Path):
    async with QBittorrentClient() as client:
        memory_code = await client.add_torrent_file(torrent_path, BTCategory.Music)
    if memory_code:
        click.echo(f"Torrent submitted to qBittorrent (tag: sprintboy_{memory_code}).")
    else:
        click.echo("Dry-run mode active in qBittorrent client — submission skipped.")


def _progress_cb(torrent: torf.Torrent, filepath: Path, pieces_done: int, pieces_total: int):
    pct = int(pieces_done / pieces_total * 100) if pieces_total else 0
    click.echo(f"\r  Hashing: {pct:3d}%", nl=False)
    if pieces_done == pieces_total:
        click.echo()


_SOURCE_CATEGORY: dict[str, BTCategory] = {
    "OPS": BTCategory.Music,
    "BTN": BTCategory.TV,
    "ANT": BTCategory.Movies,
}


@click.command()
@click.argument("torrent_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def seed_torrent(torrent_file: Path):
    """Submit a local .torrent file to qBittorrent, inferring category from its source tag.

    \b
    Source tag → category mapping:
      OPS  →  Music
      BTN  →  TV
      ANT  →  Movies
    """
    t = torf.Torrent.read(torrent_file)
    source = t.source
    if not source:
        raise click.UsageError("Torrent has no source tag — cannot determine category.")
    category = _SOURCE_CATEGORY.get(source.upper())
    if category is None:
        raise click.UsageError(
            f"Unknown source tag '{source}'. Known tags: {', '.join(_SOURCE_CATEGORY)}"
        )
    click.echo(f"Source: {source} → category: {category}")
    asyncio.run(_submit_with_category(torrent_file, category))


async def _submit_with_category(torrent_path: Path, category: BTCategory):
    async with QBittorrentClient() as client:
        memory_code = await client.add_torrent_file(torrent_path, category)
    if memory_code:
        click.echo(f"Submitted to qBittorrent (tag: sprintboy_{memory_code}).")
    else:
        click.echo("Dry-run mode active in qBittorrent client — submission skipped.")


if __name__ == "__main__":
    upload_album()
