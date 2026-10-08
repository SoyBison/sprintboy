"""Discord buttons for a "did you mean" question about a release."""

import logging
from typing import Awaitable, Callable

import discord

from bot.workflows import Choice, Pending

logger = logging.getLogger(__name__)

VIEW_TIMEOUT_SECONDS = 900
LABEL_LIMIT = 80
NONE_LABEL = "None of these"

OnPick = Callable[[discord.Interaction, Choice | None], Awaitable[None]]


class DidYouMeanView(discord.ui.View):
    """One button per option plus "None of these"; only the asker may press them."""

    def __init__(self, pending: Pending, author_id: int, on_pick: OnPick):
        super().__init__(timeout=VIEW_TIMEOUT_SECONDS)
        self.pending = pending
        self.author_id = author_id
        self.on_pick = on_pick
        self.message: discord.Message | None = None
        for option in pending.options:
            self.add_item(self._button(option.label[:LABEL_LIMIT], discord.ButtonStyle.primary, option))
        self.add_item(self._button(NONE_LABEL, discord.ButtonStyle.secondary, None))

    def _button(self, label: str, style: discord.ButtonStyle, choice: Choice | None) -> discord.ui.Button:
        button = discord.ui.Button(label=label, style=style, row=0)

        async def callback(interaction: discord.Interaction) -> None:
            await self._pressed(interaction, choice)

        button.callback = callback
        return button

    async def _pressed(self, interaction: discord.Interaction, choice: Choice | None) -> None:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("Not your request.", ephemeral=True)
            return
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(view=self)
        self.stop()
        await self.on_pick(interaction, choice)

    async def on_timeout(self) -> None:
        if self.message is None:
            return
        try:
            await self.message.edit(view=None)
        except discord.HTTPException:
            logger.debug("Could not remove the expired buttons", exc_info=True)
