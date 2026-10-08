import asyncio

import click

from bot.config import Config, setup_logging
from bot.tools import TorrentContext


@click.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--model", "-m", default=None, help="Override the configured model")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "ollama"]),
    default=None,
    help="Override LLM_PROVIDER for this run",
)
@click.option("--verbose", "-v", is_flag=True, help="Log every tool call and result")
def ask(query: tuple[str, ...], model: str | None, provider: str | None, verbose: bool):
    """Run one query through the agent without going via Discord.

    Handy for checking whether a self-hosted model can actually drive the tools:
    the agent really does search and add torrents, so use a read-only query or
    set QBITTORRENT_DRY_RUN.
    """
    asyncio.run(_ask(" ".join(query), model, provider, verbose))


async def _ask(
    query: str, model: str | None, provider: str | None, verbose: bool = False
):
    setup_logging("DEBUG" if verbose else "WARNING")
    if provider:
        Config.LLM_PROVIDER = provider
    if model:
        if Config.LLM_PROVIDER == "ollama":
            Config.OLLAMA_MODEL = model
        else:
            Config.ANTHROPIC_MODEL = model

    # Imported here so that --provider is applied before the model is built.
    from bot import turn
    from bot.decide import drain_shadow

    context = TorrentContext(
        search_results={}, internal_torrents={}, torrent_types=set()
    )

    click.echo(
        click.style(
            f"[{Config.LLM_PROVIDER}:"
            f"{Config.OLLAMA_MODEL if Config.LLM_PROVIDER == 'ollama' else Config.ANTHROPIC_MODEL}]",
            fg="cyan",
        )
    )
    from bot.output import add_nudge

    agent, messages, route, _run_id = await turn.prepare(
        [{"role": "user", "content": query}]
    )
    if route is None:
        click.echo(click.style("route: none", fg="cyan"))
    else:
        tools = [t.name for t in turn.select_tools(route)]
        click.echo(
            click.style(
                f"route: {route.domain}/{route.kind} "
                f"({route.domain_p:.2f}/{route.kind_p:.2f}) "
                f"count={route.count} tools={tools}",
                fg="cyan",
            )
        )
    messages = await _stream(agent, messages, context)
    # Same single retry the Discord bot makes for a search that was never added.
    nudge = add_nudge(messages, list(context.internal_torrents), route)
    if nudge:
        click.echo(click.style("  (nudging: download turn added nothing)", fg="magenta"))
        await _stream(agent, [*messages, {"role": "user", "content": nudge}], context)
    await drain_shadow()


async def _stream(agent, messages, context) -> list:
    """Run the agent, echoing tool calls and replies, and return the final messages."""
    from bot.output import clean_reply, message_text

    final = messages
    async for mode, chunk in agent.astream(
        {"messages": messages},
        context=context,
        stream_mode=["updates", "values"],
    ):
        if mode == "values":
            final = chunk.get("messages", final)
            continue
        for node, update in chunk.items():
            for message in (
                update.get("messages", []) if isinstance(update, dict) else []
            ):
                for call in getattr(message, "tool_calls", None) or []:
                    click.echo(
                        click.style(f"  → {call['name']}({call['args']})", fg="yellow")
                    )
                if getattr(message, "type", None) == "tool":
                    body = message_text(message.content)
                    click.echo(click.style(f"  ← {body[:400]}", fg="green"))
                elif node == "model" and clean_reply(message.content):
                    click.echo(clean_reply(message.content))
    return final

if __name__ == "__main__":
    ask()
