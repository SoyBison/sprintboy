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
    from bot.main import SYSTEM_PROMPT, AGENT_TOOLS, build_llm
    from bot.output import clean_reply, message_text
    from langchain.agents import create_agent

    agent = create_agent(build_llm(), tools=AGENT_TOOLS, context_schema=TorrentContext)
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
    async for chunk in agent.astream(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
        },
        context=context,
        stream_mode="updates",
    ):
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


if __name__ == "__main__":
    ask()
