"""Housekeeping on the Plex server, mostly clearing up after the test suite.

`just test` runs the live integration tests, and test_make_playlist really does
create a playlist. Those built up unnoticed -- six empty "Rapp Snitch Knishes"
playlists by the time anyone looked -- so clearing them is a command rather
than something to do by hand in the Plex UI.
"""

import asyncio

import click

from bot.config import setup_logging
from bot.netcode import TEST_PLAYLIST_PREFIX, PlexAPIClient


@click.command()
@click.option(
    "--prefix",
    default=TEST_PLAYLIST_PREFIX,
    show_default=True,
    help="Delete playlists whose title starts with this.",
)
@click.option(
    "--no-legacy",
    is_flag=True,
    help="Leave the empty playlists the older tests named after a track.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Actually delete. Without this the command only lists what it found.",
)
@click.option("--verbose", "-v", is_flag=True, help="Log every Plex request")
def clear_test_playlists(prefix: str, no_legacy: bool, yes: bool, verbose: bool):
    """List, and with --yes delete, the playlists the tests leave on Plex."""
    asyncio.run(_clear(prefix, not no_legacy, yes, verbose))


async def _clear(prefix: str, include_legacy: bool, confirmed: bool, verbose: bool):
    setup_logging("DEBUG" if verbose else "WARNING")
    async with PlexAPIClient() as plex:
        found = await plex.find_test_playlists(prefix, include_legacy=include_legacy)
        if not found:
            click.echo("No test playlists on Plex.")
            return

        click.echo(f"{len(found)} test playlist(s) on Plex:")
        for playlist in found:
            click.echo(f"  {playlist.id}  {playlist.title}  ({playlist.items} items)")

        if not confirmed:
            click.echo(
                click.style(
                    "Nothing deleted. Re-run with --yes to delete these.", fg="yellow"
                )
            )
            return

        deleted = await plex.delete_playlists(found)
        click.echo(
            click.style(f"Deleted {deleted} of {len(found)} playlist(s).", fg="green")
        )


if __name__ == "__main__":
    clear_test_playlists()
