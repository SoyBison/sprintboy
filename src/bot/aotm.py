"""Offer Orpheus' Album of the Month to the owner over Discord DM."""

import asyncio
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord.ext import tasks

from bot.config import Config
from bot.netcode import BTCategory, QBittorrentClient
from bot.orpheus import OrpheusClient, current_pick

logger = logging.getLogger(__name__)

# Set by setup(): awaited with (message, torrent name, memory code) once a grab
# has been added to qBittorrent.
_on_added = None


def load_state() -> dict:
    try:
        with open(Config.AOTM_STATE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_state(state: dict) -> None:
    path = Path(Config.AOTM_STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


async def _is_owner(interaction: discord.Interaction) -> bool:
    if Config.DISCORD_OWNER_ID:
        return interaction.user.id == int(Config.DISCORD_OWNER_ID)
    return await interaction.client.is_owner(interaction.user)


class AotmButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"aotm:(?P<action>grab|skip):(?P<torrent_id>\d+):(?P<key>\d+)",
):
    def __init__(self, action: str, torrent_id: int, key: str):
        self.action = action
        self.torrent_id = int(torrent_id)
        self.key = str(key)
        grab = action == "grab"
        super().__init__(
            discord.ui.Button(
                label="Grab" if grab else "Skip",
                style=discord.ButtonStyle.success if grab else discord.ButtonStyle.secondary,
                custom_id=f"aotm:{action}:{self.torrent_id}:{self.key}",
            )
        )

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["torrent_id"]), match["key"])

    async def callback(self, interaction: discord.Interaction):
        if not await _is_owner(interaction):
            await interaction.response.send_message("Not for you.", ephemeral=True)
            return
        state = load_state()
        entry = state.get(self.key, {})
        if entry.get("status") in ("grabbed", "skipped"):
            await interaction.response.send_message("Already handled.", ephemeral=True)
            return
        if self.action == "skip":
            entry["status"] = "skipped"
            state[self.key] = entry
            save_state(state)
            await interaction.response.edit_message(
                content=f"{interaction.message.content}\n\nSkipped.", view=None
            )
        else:
            await self._grab(interaction, state, entry)

    async def _grab(self, interaction: discord.Interaction, state: dict, entry: dict):
        await interaction.response.defer()
        try:
            async with OrpheusClient() as client:
                data = await client.download_torrent(self.torrent_id)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"{self.torrent_id}.torrent"
                path.write_bytes(data)
                async with QBittorrentClient() as q:
                    memory_code = await q.add_torrent_file(path, BTCategory.Music)
                    if memory_code is None:
                        await interaction.followup.send("Dry run is on, nothing was added.")
                        return
                    try:
                        info = await q.get_torrent_info(memory_code, timeout=20)
                    except TimeoutError:
                        await interaction.followup.send(
                            "qBittorrent never showed the torrent, so it was not added."
                        )
                        return
        except Exception as e:
            logger.exception("AoTM grab failed")
            await interaction.followup.send(f"Couldn't grab it: {e}")
            return

        entry["status"] = "grabbed"
        state[self.key] = entry
        save_state(state)
        message = interaction.message
        await message.edit(content=f"{message.content}\n\nGrabbing it.", view=None)
        if _on_added:
            await _on_added(message, info.name, memory_code)


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


async def check_and_ask(bot, owner) -> str:
    if not Config.ORPHEUS_API_KEY:
        return "no key"
    async with OrpheusClient() as client:
        current = await current_pick(client)
    if current is None:
        return "no current pick"
    pick, torrent = current
    state = load_state()
    if pick.key in state:
        return "already asked"
    until = f"{pick.freeleech_until:%b} {pick.freeleech_until.day}"

    if torrent is None:
        await owner.send(
            f"AoTM {pick.round}: {pick.artist} - {pick.album} is freeleech until "
            f"{until}, but I couldn't find a freeleech FLAC torrent for it."
        )
        state[pick.key] = {
            "status": "asked",
            "torrent_id": 0,
            "label": f"{pick.artist} - {pick.album}",
            "asked_at": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state)
        return "asked"

    from bot.tools import check_for_album

    result = str(await check_for_album.ainvoke({"artist": pick.artist, "title": pick.album}))
    note = ""
    if "has no albums" in result:
        note = (
            f"You don't seem to have anything by {pick.artist} "
            f"(the check can miss artists filed under another name).\n"
        )
    elif "COLLISION" in result:
        note = "Looks like you may already have it.\n"

    content = (
        f"**Album of the Month** ({pick.round}) is freeleech until {until}:\n"
        f"{torrent.label}, {torrent.seeders} seeders, {_human_size(torrent.size)}\n"
        f"{note}Want it?"
    )
    view = discord.ui.View(timeout=None)
    view.add_item(AotmButton("grab", torrent.torrent_id, pick.key))
    view.add_item(AotmButton("skip", torrent.torrent_id, pick.key))
    message = await owner.send(content, view=view)
    state[pick.key] = {
        "status": "asked",
        "torrent_id": torrent.torrent_id,
        "label": torrent.label,
        "asked_at": datetime.now(timezone.utc).isoformat(),
        "message_id": message.id,
    }
    save_state(state)
    return "asked"


def setup(bot, on_added):
    """Register the persistent buttons and return the (unstarted) check loop."""
    global _on_added
    _on_added = on_added
    bot.add_dynamic_items(AotmButton)

    @tasks.loop(hours=Config.AOTM_CHECK_HOURS)
    async def aotm_loop():
        try:
            if Config.DISCORD_OWNER_ID:
                owner = await bot.fetch_user(int(Config.DISCORD_OWNER_ID))
            else:
                owner = (await bot.application_info()).owner
            logger.info(f"AoTM check: {await check_and_ask(bot, owner)}")
        except Exception:
            logger.exception("AoTM check failed")

    @aotm_loop.before_loop
    async def _wait():
        await bot.wait_until_ready()

    return aotm_loop


async def _cli() -> None:
    if not Config.ORPHEUS_API_KEY:
        print("ORPHEUS_API_KEY is not set")
        return
    async with OrpheusClient() as client:
        current = await current_pick(client)
    if current is None:
        print("No current Album of the Month (none within its freeleech window).")
        return
    pick, torrent = current
    print(f"AoTM {pick.round}: {pick.artist} - {pick.album}")
    print(f"Announced {pick.announced:%Y-%m-%d}, freeleech until {pick.freeleech_until:%Y-%m-%d}")
    if torrent is None:
        print("No freeleech FLAC torrent found.")
    else:
        print(f"Would offer: {torrent.label} (id {torrent.torrent_id}, "
              f"{torrent.seeders} seeders, {_human_size(torrent.size)})")


if __name__ == "__main__":
    asyncio.run(_cli())
