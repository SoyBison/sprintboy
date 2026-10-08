import asyncio
import time

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
    from bot.output import clean_reply
    from bot.runlog import record_run

    started = time.perf_counter()
    t = await turn.prepare([{"role": "user", "content": query}])
    if t.route is None:
        click.echo(click.style("route: none", fg="cyan"))
    else:
        route = t.route
        tools = [tool.name for tool in t.tools]
        click.echo(
            click.style(
                f"route: {route.domain}/{route.kind} "
                f"({route.domain_p:.2f}/{route.kind_p:.2f}) "
                f"count={route.count} tools={tools}",
                fg="cyan",
            )
        )

    def echo(event: str, data: dict) -> None:
        if event == "tool_call":
            click.echo(click.style(f"  → {data['name']}({data['args']})", fg="yellow"))
        elif event == "tool_result":
            click.echo(click.style(f"  ← {data['result'][:400]}", fg="green"))
        elif event == "reply":
            reply = clean_reply(data["content"])
            if reply:
                click.echo(reply)

    result = await turn.run(t, context, on_event=echo)
    for pending in result.pending:
        options = "  ".join(f"{i}) {o.label}" for i, o in enumerate(pending.options, 1))
        click.echo(click.style(f'  did you mean "{pending.wanted}": {options}', fg="magenta"))
    if result.nudged:
        click.echo(click.style("  (nudged: download turn added nothing)", fg="magenta"))
    try:
        from bot.output import best_reply, unfulfilled_note

        new_torrents = list(context.internal_torrents)
        reply = best_reply(result.messages, new_torrents)
        note = unfulfilled_note(result.messages, new_torrents)
        record_run(
            run_id=t.run_id,
            message_id=None,
            conversation_id=None,
            author="cli",
            text=query,
            route=t.route,
            result=result,
            new_torrents=new_torrents,
            reply=reply,
            note=note,
            seconds=time.perf_counter() - started,
        )
    except Exception as e:
        click.echo(click.style(f"Could not write the run log: {e}", fg="red"))
    await drain_shadow()


if __name__ == "__main__":
    ask()
