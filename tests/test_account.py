from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot import turn, workflows
from bot.choices import ConfirmView
from bot.config import Config
from bot.decide import Decision
from bot.orpheus import (
    AccountStats,
    OrpheusClient,
    OrpheusError,
    ShopItem,
    parse_token_shop,
    plan_purchase,
)
from bot.routing import Route

INDEX = {
    "username": "me",
    "id": 1,
    "authkey": "SECRETAUTH",
    "passkey": "SECRETPASS",
    "userstats": {
        "uploaded": 10, "downloaded": 20, "ratio": 0.61, "requiredratio": 0.6,
        "bonusPoints": 1874, "bonusPointsPerHour": 149.76, "tokens": 6, "class": "User",
    },
}

SHOP = """
<form method="post"><input type="hidden" name="auth" value="x"><table>
<tr><td>1 Freeleech Token</td><td style="text-align:right">1,000</td>
<td><input type="submit" name="token-1" value="Purchase!"></td></tr>
<tr><td>10 Freeleech Tokens</td><td style="text-align:right">9,000</td>
<td><a href="bonus.php?action=prepare&item=token-10">Prepare</a></td></tr>
<tr><td>2 Freeleech Tokens</td><td style="text-align:right">1,900</td>
<td><input type="submit" name="token-2" value="Purchase!"></td></tr>
<tr><td>Gift 1 Token</td><td style="text-align:right">1,500</td>
<td><input type="submit" name="other-gift-1" value="Purchase!"></td></tr>
</table></form>
"""


def stats(**kw):
    base = dict(username="me", uploaded=1, downloaded=1, ratio=1.5, required_ratio=0.6,
                bonus_points=5000, bonus_per_hour=100.0, tokens=6, user_class="User")
    return AccountStats(**{**base, **kw})


@pytest.mark.asyncio
async def test_account_parsing_hides_secrets():
    client = OrpheusClient()
    client._json = AsyncMock(return_value=INDEX)
    s, authkey = await client.account()
    assert (s.tokens, s.bonus_points, s.ratio, s.required_ratio) == (6, 1874, 0.61, 0.6)
    assert authkey == "SECRETAUTH"
    assert "SECRET" not in repr(s)


@pytest.mark.asyncio
async def test_token_shop_parsing_and_login_detection():
    client = OrpheusClient()
    client._web = AsyncMock(return_value=(200, {}, SHOP))
    items = await client.token_shop()
    assert [(i.label, i.tokens, i.price) for i in items] == [
        ("token-1", 1, 1000), ("token-10", 10, 9000), ("token-2", 2, 1900),
    ]
    login = '<input name="username"><input name="password">'
    client._web = AsyncMock(return_value=(200, {}, login))
    with pytest.raises(OrpheusError, match="session cookie"):
        await client.token_shop()
    client._web = AsyncMock(return_value=(302, {"Location": "login.php"}, ""))
    with pytest.raises(OrpheusError, match="session cookie"):
        await client.token_shop()


@pytest.mark.asyncio
async def test_buy_success_and_failure():
    client = OrpheusClient()
    client._web = AsyncMock(return_value=(302, {"Location": "bonus.php?complete=token-1"}, ""))
    await client.buy("token-1", "k")
    client._web.assert_awaited_with(
        "POST", "bonus.php", {"auth": "k", "action": "purchase", "label": "token-1"}
    )
    client._web = AsyncMock(return_value=(400, {}, "<p>Not enough <b>funds</b> due to lack of funds</p>"))
    with pytest.raises(OrpheusError, match="lack of funds"):
        await client.buy("token-1", "k")


def test_plan_purchase():
    items = parse_token_shop(SHOP)
    one = ShopItem("token-1", "1 Freeleech Token", 1, 1000)
    ten = ShopItem("token-10", "10 Freeleech Tokens", 10, 9000)
    plan = plan_purchase([one, ten], 11, 20000)
    assert sum(i.price for i in plan) == 10000 and sum(i.tokens for i in plan) == 11
    assert plan_purchase([one, ten], 11, 5000) == []
    assert plan_purchase(items, None, 100) == []


def test_plan_purchase_max_mode():
    one = ShopItem("token-1", "1", 1, 1000)
    ten = ShopItem("token-10", "10", 10, 9000)
    best = plan_purchase([one, ten], None, 9500)
    assert sum(i.tokens for i in best) == 10


class FakeClient:
    def __init__(self, st=None, items=None, buy_error_at=None):
        self.st = st or stats()
        self.items = items if items is not None else parse_token_shop(SHOP)
        self.bought = []
        self.buy_error_at = buy_error_at

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def account(self):
        return self.st, "AUTH"

    async def token_shop(self):
        return self.items

    async def buy(self, label, authkey):
        if self.buy_error_at is not None and len(self.bought) == self.buy_error_at:
            raise OrpheusError("lack of funds")
        self.bought.append((label, authkey))
        self.st = stats(tokens=self.st.tokens + 1, bonus_points=self.st.bonus_points - 1000)


def decision(intent, mx=0.0):
    return Decision("t", "m", {"intent": {"choice": intent, "probabilities": {intent: 0.9}},
                               "max": {"noul": mx}}, 0.0)


async def run_account(text, fake, intent, mx=0.0, cookie="cookie"):
    # Stay patched so a confirmation's action can run later; stopall() cleans up.
    patch("bot.workflows.decide", AsyncMock(return_value=decision(intent, mx))).start()
    patch("bot.workflows.OrpheusClient", lambda: fake).start()
    patch.object(Config, "ORPHEUS_SESSION_COOKIE", cookie).start()
    return await workflows.account(text)


@pytest.fixture(autouse=True)
def _stop_patches():
    yield
    patch.stopall()


@pytest.mark.asyncio
async def test_status_and_ratio_warning():
    r = await run_account("tokens?", FakeClient(), "status")
    assert r.reply == (
        "You have 6 freeleech tokens and 5,000 bonus points (+100/hour, about 2,400/day). "
        "Ratio 1.50 (you need 0.60)."
    )
    assert r.confirm is None
    r = await run_account("ratio?", FakeClient(stats(ratio=0.61)), "status")
    assert "close to ratio watch" in r.reply


@pytest.mark.asyncio
async def test_buy_without_cookie():
    r = await run_account("buy 2 tokens", FakeClient(), "buy", cookie="")
    assert "ORPHEUS_SESSION_COOKIE" in r.reply and r.confirm is None


@pytest.mark.asyncio
async def test_buy_confirm_runs_buys_then_reports():
    fake = FakeClient()
    r = await run_account("buy 2 freeleech tokens", fake, "buy")
    assert r.confirm is not None and fake.bought == []
    assert "Buy 2 freeleech tokens for 1,900 bonus points? You'd have 3,100 left." in r.reply
    reply = await r.confirm.action()
    assert fake.bought == [("token-2", "AUTH")]
    assert reply == "Bought 2 tokens. You now have 7 tokens and 4,000 bonus points."


@pytest.mark.asyncio
async def test_buy_partial_failure():
    fake = FakeClient(buy_error_at=1)
    r = await run_account("buy 3 tokens", fake, "buy")
    assert "Buy 3" in r.reply
    reply = await r.confirm.action()
    assert reply.startswith("Bought ") and "of 3 before Orpheus said: lack of funds" in reply


@pytest.mark.asyncio
async def test_buy_max_and_insufficient():
    r = await run_account("buy as many as I can", FakeClient(), "buy", mx=0.9)
    assert "Buy 5 freeleech tokens" in r.reply or "Buy 4" in r.reply
    r = await run_account("buy 10 tokens", FakeClient(stats(bonus_points=500)), "buy")
    assert "You can't afford 10 freeleech tokens: the cheapest is 9,000 points for 10." in r.reply
    r = await run_account("buy tokens", FakeClient(stats(bonus_points=500)), "buy")
    assert "can't afford 1 freeleech token: the cheapest is 1,000 points for 1." in r.reply
    r = await run_account("buy as many as I can", FakeClient(stats(bonus_points=500)), "buy", mx=0.9)
    assert "can't afford any tokens yet: the cheapest is 1,000 points for 1." in r.reply


def make_interaction(user_id):
    i = MagicMock()
    i.user = SimpleNamespace(id=user_id)
    i.response.send_message = AsyncMock()
    i.response.edit_message = AsyncMock()
    i.followup.send = AsyncMock()
    return i


@pytest.mark.asyncio
async def test_confirm_view_author_only_and_single_run():
    action = AsyncMock(return_value="Bought it")
    view = ConfirmView(workflows.Confirmation("p", action), author_id=1)
    confirm, cancel = view.children
    stranger = make_interaction(2)
    await confirm.callback(stranger)
    stranger.response.send_message.assert_awaited_with("Not your request.", ephemeral=True)
    action.assert_not_awaited()
    owner = make_interaction(1)
    await confirm.callback(owner)
    owner.followup.send.assert_awaited_with("Bought it")
    again = make_interaction(1)
    await confirm.callback(again)
    await cancel.callback(make_interaction(1))
    action.assert_awaited_once()
    assert all(c.disabled for c in view.children)


@pytest.mark.asyncio
async def test_confirm_view_cancel():
    action = AsyncMock()
    view = ConfirmView(workflows.Confirmation("p", action), author_id=1)
    inter = make_interaction(1)
    await view.children[1].callback(inter)
    inter.followup.send.assert_awaited_with("Cancelled, nothing bought.")
    action.assert_not_awaited()


@pytest.mark.asyncio
async def test_turn_dispatches_tracker_to_account():
    t = turn.Turn(tools=[], messages=[], route=Route("tracker", 0.9, "question", 0.9, None),
                  run_id="r", text="my ratio")
    wf = workflows.WorkflowResult(reply="hi", steps=[])
    with patch("bot.turn.workflows.account", AsyncMock(return_value=wf)) as m:
        result = await turn.run(t, MagicMock(internal_torrents={}))
    m.assert_awaited_once_with("my ratio", run_id="r")
    assert result.stopped == "workflow"


@pytest.mark.asyncio
async def test_token_words_route_to_the_tracker(monkeypatch):
    from unittest.mock import AsyncMock

    from bot import routing
    from bot.decide import Decision

    answers = {
        "domain": {"type": "choice", "choice": "chat", "confidence": 0.9,
                   "probabilities": {"chat": 0.95, "tracker": 0.03, "music": 0.02}},
        "kind": {"type": "choice", "choice": "question", "confidence": 1.0,
                 "probabilities": {"question": 1.0}},
    }
    monkeypatch.setattr(routing, "decide", AsyncMock(return_value=Decision("jev", "j", answers, 0.1)))
    r = await routing.route("how many tokens do I have left?")
    assert r.domain == "tracker" and r.domain_p >= routing.MIN_DOMAIN_CONFIDENCE
    r = await routing.route("thanks!")
    assert r.domain == "chat"
