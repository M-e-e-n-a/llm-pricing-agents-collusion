"""Offline tests for pilot/fork.py (no network): python -m pytest tests/test_fork.py"""

import json

import pytest

from pilot.backends import FakeBackend
from pilot.fork import (
    DEFAULT_VERSIONS,
    FORK_CAP,
    FORK_MIN,
    HOLD,
    POST_ROUNDS,
    REVEAL_AT,
    SETTLE_SPAN,
    VERSIONS,
    ScriptedRival,
    SessionSpec,
    _repair,
    branches,
    load_texts,
    make_run,
    run_forks,
    settled,
)
from pilot.market import parse

TEXTS = load_texts()


class ConstantBackend:
    """A always answers the same price, so its prices settle at once."""

    def __init__(self, price=1.5):
        self.price, self.calls = price, 0

    def describe(self):
        return {"name": "constant", "price": self.price}

    def generate(self, batch, seeds):
        self.calls += 1
        out = []
        for messages in batch:
            rendered = "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)
            answer = {"observations": "o", "plans": "p", "insights": "i", "chosen_price": self.price,
                      "message_to_competitor": "ok"}
            out.append((rendered, json.dumps(answer)))
        return out


class CrashingBackend(FakeBackend):
    """FakeBackend that dies on its `crash_at`-th call, like a lost server mid-run."""

    def __init__(self, crash_at):
        super().__init__()
        self.crash_at, self.calls = crash_at, 0

    def generate(self, batch, seeds):
        self.calls += 1
        if self.calls == self.crash_at:
            raise RuntimeError("server gone")
        return super().generate(batch, seeds)


def quiet(_):
    pass


def calls_by(out_dir):
    with open(out_dir / "calls.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {(r["session_id"], r["version"], r["round"]): r["prompt"] for r in rows if r["attempt"] == 1}


def test_best_response():
    rival = ScriptedRival(make_run(SessionSpec("s", "plain", 0), TEXTS, 100, 10), TEXTS, 0, 0.0)
    assert rival.best_response(1.90) == pytest.approx(1.6526, abs=2e-3)
    nash = float(rival.run.env.nash_prices[0])
    assert rival.best_response(nash) == pytest.approx(nash, abs=2e-3)


def test_settled():
    assert settled([1.5, 1.55, 1.5, 1.5 + SETTLE_SPAN, 1.52])  # exactly the span still counts, despite float error
    assert not settled([1.5, 1.55, 1.5, 1.5 + SETTLE_SPAN + 0.01, 1.52])
    assert not settled([1.5, 1.5, 1.5, 1.5])  # fewer than 5 rounds


def test_fork_round_and_windows(tmp_path):
    results = run_forks(ConstantBackend(1.5), [SessionSpec("plain_s0", "plain", 0)], TEXTS, 100, tmp_path, log=quiet)
    assert sorted(r["version"] for r in results) == sorted(DEFAULT_VERSIONS)
    for r in results:
        assert r["t0"] == FORK_MIN and r["settled"] is True
        assert len(r["a_prices"]) == len(r["b_prices"]) == FORK_MIN + POST_ROUNDS
        assert r["median_price"] == 1.5 and not any(r["accepted"].values())
        assert r["pre_fork_median"] == 1.5 and r["eligible"] == {"below_threshold": True, "below_best_response": True}
        assert r["best_response_to_hold"] == pytest.approx(1.6526, abs=2e-3)
        b_after = r["b_prices"][FORK_MIN:]
        if VERSIONS[r["version"]].hold:
            assert b_after == [HOLD] * POST_ROUNDS
        else:
            assert HOLD not in b_after  # N keeps best-responding
        assert r["b_prices"][:FORK_MIN] == results[0]["b_prices"][:FORK_MIN]  # shared warm-up


def test_session_already_above_threshold_is_not_eligible(tmp_path):
    results = run_forks(ConstantBackend(1.8), [SessionSpec("plain_s0", "plain", 0)], TEXTS, 100, tmp_path, log=quiet)
    assert {r["pre_fork_median"] for r in results} == {1.8}
    assert all(r["eligible"] == {"below_threshold": False, "below_best_response": False} for r in results)


def test_unsettled_session_forks_at_cap(tmp_path):
    results = run_forks(FakeBackend(bad_rate=0.0), [SessionSpec("plain_s0", "plain", 0)], TEXTS, 100, tmp_path,
                        log=quiet)
    assert {r["t0"] for r in results} == {FORK_CAP} and {r["settled"] for r in results} == {False}


def test_versions_differ_only_where_intended(tmp_path):
    run_forks(ConstantBackend(1.5), [SessionSpec("plain_s1", "plain", 1)], TEXTS, 100, tmp_path, log=quiet)
    p = calls_by(tmp_path)
    t0, sid = FORK_MIN, "plain_s1"
    offer, offer_news, neutral = TEXTS["offers"][1], TEXTS["offer_news"][1], TEXTS["neutral_news"][1]
    # Round t0 + 1: A has not yet seen B's fork move, so every version's prompt is the same
    assert len({p[(sid, v, t0 + 1)] for v in DEFAULT_VERSIONS}) == 1
    # Round t0 + 2: O differs from R in exactly one line, B's chat message, which is the offer
    r_lines, o_lines = p[(sid, "R", t0 + 2)].splitlines(), p[(sid, "O", t0 + 2)].splitlines()
    diff = [(a, b) for a, b in zip(r_lines, o_lines) if a != b]
    assert len(r_lines) == len(o_lines) and len(diff) == 1 and offer in diff[0][1]
    assert offer not in p[(sid, "R", t0 + 2)]
    # O_news carries the offer only inside the quoted news item, never as B's chat message
    o_news = p[(sid, "O_news", t0 + 2)]
    assert o_news.count(offer) == 1 and offer in offer_news and offer_news in o_news
    # News appears from t0 + 2 on, only in the news versions; the offer news reuses the offer's proposal
    assert "NEWS" not in p[(sid, "O_news", t0 + 1)]
    assert offer_news in p[(sid, "O_news", t0 + 2)] and offer_news in p[(sid, "O_news", t0 + POST_ROUNDS)]
    assert neutral in p[(sid, "R_news", t0 + 2)] and offer_news not in p[(sid, "R_news", t0 + 2)]
    assert "NEWS" not in p[(sid, "R", t0 + 2)] and "NEWS" not in p[(sid, "O", t0 + 2)]
    # R and R_news differ only by the news section
    r_news = p[(sid, "R_news", t0 + 2)]
    assert r_news.replace(r_news[r_news.index("Section name: NEWS"):r_news.index("Now you have all")], "") == \
        p[(sid, "R", t0 + 2)]


def test_approach_versions_swap_only_the_chat_line(tmp_path):
    run_forks(ConstantBackend(1.5), [SessionSpec("plain_s1", "plain", 1)], TEXTS, 100, tmp_path, log=quiet)
    p = calls_by(tmp_path)
    t0, sid = FORK_MIN, "plain_s1"
    r_lines = p[(sid, "R", t0 + 2)].splitlines()
    for version, key in (("Announce", "announce"), ("Hint", "hint"), ("Benign", "benign")):
        lines = p[(sid, version, t0 + 2)].splitlines()
        diff = [(a, b) for a, b in zip(r_lines, lines) if a != b]
        assert len(lines) == len(r_lines) and len(diff) == 1 and TEXTS[key][1] in diff[0][1]


def test_repeat_offer_sends_the_chat_text_every_round(tmp_path):
    t0 = FORK_MIN
    spec = SessionSpec("plain_s1", "plain", 1)
    for repeat in (False, True):
        b = ScriptedRival(make_run(spec, TEXTS, 100, 10), TEXTS, 1, 0.02, repeat)
        for name in ("O", "Announce", "Hint", "Benign"):
            text = TEXTS[VERSIONS[name].chat][1]
            sent = [r for r in range(t0 + 1, t0 + POST_ROUNDS + 1)
                    if b.move(r, 1.5, VERSIONS[name], t0)["message_to_competitor"] == text]
            assert sent == (list(range(t0 + 1, t0 + POST_ROUNDS + 1)) if repeat else [t0 + 1])
        assert all(TEXTS["offers"][1] != b.move(r, 1.5, VERSIONS["R"], t0)["message_to_competitor"]
                   for r in range(t0 + 1, t0 + POST_ROUNDS + 1))


def test_versions_subset_and_repeat_are_recorded(tmp_path):
    spec = [SessionSpec("plain_s1", "plain", 1)]
    results = run_forks(ConstantBackend(1.5), spec, TEXTS, 100, tmp_path, log=quiet, versions=["R", "O"], repeat=True)
    assert sorted(r["version"] for r in results) == ["O", "R"]
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["repeat_offer"] is True and [v["name"] for v in cfg["versions"]] == ["R", "O"]
    with pytest.raises(AssertionError):  # resuming with a different repeat setting is refused
        run_forks(ConstantBackend(1.5), spec, TEXTS, 100, tmp_path, log=quiet, versions=["R", "O"])


def test_branches_do_not_share_state_and_seeds():
    spec = SessionSpec("plain_s0", "plain", 0)
    run = make_run(spec, TEXTS, 100, 10)
    snapshot = {"t0": 20, "settled": True, "history": run.history}
    units = branches(spec, snapshot, list(VERSIONS), TEXTS, 100, 10, 0.02)
    units[0].run.history["Firm A"][99] = {"chosen_price": 9.0}
    assert all(99 not in u.run.history["Firm A"] for u in units[1:]) and 99 not in run.history["Firm A"]
    assert len({id(u.run.env) for u in units}) == len(units)
    seeds = {u.version.name: u.seed(1) for u in units}
    assert len({seeds[v] for v in VERSIONS if v != "R_again"}) == 1 and seeds["R"] != seeds["R_again"]
    assert units[0].seed(1) != units[0].seed(2)


def test_resume_after_crash_matches_clean_run(tmp_path):
    sessions = [SessionSpec("plain_s0", "plain", 0), SessionSpec("follower_s1", "follower", 1)]
    clean = run_forks(FakeBackend(), sessions, TEXTS, 100, tmp_path / "clean", log=quiet)
    with pytest.raises(RuntimeError):  # 35 warm-up rounds, then dies during the branches
        run_forks(CrashingBackend(crash_at=50), sessions, TEXTS, 100, tmp_path / "resumed", log=quiet)
    forks = (tmp_path / "resumed" / "forks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(forks) == 2  # the crash came after both forks, so the resume rebuilds branches from saved state
    with open(tmp_path / "resumed" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write('{"session_id": "plain_s0", "ver')  # a write cut off by the crash
    resumed = run_forks(FakeBackend(), sessions, TEXTS, 100, tmp_path / "resumed", log=quiet)
    key = lambda r: (r["session_id"], r["version"])
    assert len(resumed) == len(clean) == 2 * len(DEFAULT_VERSIONS)
    assert sorted(resumed, key=key) == sorted(clean, key=key)
    cfg = json.loads((tmp_path / "resumed" / "config.json").read_text(encoding="utf-8"))
    assert cfg["start"] == 2


def test_parse_survives_bad_escape():
    # Live gpt-oss reply shape that crashed the parser on 24 Oct: a backslash escape JSON does not allow
    text = '{\n  "observations": "Gap \\$0.12 \\(small\\)",\n  "plans": "p",\n  "insights": "i",\n' \
           '  "chosen_price": 1.74,\n  "message_to_competitor": "ok"\n}'
    parsed = parse(text, chat=True)
    assert parsed["chosen_price"] == 1.74 and "0.12" in parsed["observations"]


def test_repair_drops_partial_line(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n{"b": ', encoding="utf-8")
    _repair(f)
    assert f.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_resume_refuses_changed_texts(tmp_path):
    run_forks(ConstantBackend(), [SessionSpec("plain_s0", "plain", 0)], TEXTS, 100, tmp_path, log=quiet)
    changed = TEXTS | {"bland": ["something else"]}
    with pytest.raises(AssertionError):
        run_forks(ConstantBackend(), [SessionSpec("plain_s0", "plain", 0)], changed, 100, tmp_path, log=quiet)


def test_parser_crash_is_logged_and_retried(tmp_path, monkeypatch):
    from pilot import fork

    real, seen = fork.parse, []

    def flaky(text, chat):  # the first reply breaks the parser, like the bad escape in the 24 Oct live check
        seen.append(text)
        if len(seen) == 1:
            raise ValueError("boom")
        return real(text, chat)

    monkeypatch.setattr(fork, "parse", flaky)
    results = run_forks(ConstantBackend(1.5), [SessionSpec("plain_s0", "plain", 0)], TEXTS, 100, tmp_path, log=quiet)
    assert len(results) == len(DEFAULT_VERSIONS)
    with open(tmp_path / "calls.jsonl", encoding="utf-8") as f:
        first = json.loads(f.readline())
    assert first["parse_error"] == "ValueError: boom" and first["completion"] == seen[0] and not first["parsed"]


# Disruption versions (prereg-v1.1)
DTEXTS = TEXTS | {"warn_news": ["WARNING: undercover pricing agents."], "reveal": ["REVEAL: I am a compliance agent."]}


def without_news(prompt):
    return prompt.replace(prompt[prompt.index("Section name: NEWS"):prompt.index("Now you have all")], "")


def test_default_versions_are_the_nine_of_prereg_v1():
    assert DEFAULT_VERSIONS == ["N", "R", "R_again", "O", "O_news", "R_news", "Announce", "Hint", "Benign"]


def test_disruption_versions_differ_only_where_intended(tmp_path):
    spec = [SessionSpec("plain_s1", "plain", 1)]
    results = run_forks(ConstantBackend(1.5), spec, DTEXTS, 100, tmp_path, log=quiet, repeat=True,
                        versions=["R", "O", "O_warn", "R_warn", "O_reveal"])
    assert all(r["b_prices"][FORK_MIN:] == [HOLD] * POST_ROUNDS for r in results)
    p = calls_by(tmp_path)
    t0, sid, warn, reveal = FORK_MIN, "plain_s1", DTEXTS["warn_news"][0], DTEXTS["reveal"][0]
    for r in range(t0 + 2, t0 + POST_ROUNDS + 1):
        # the warning versions are O and R plus the news section, nothing else
        assert warn in p[(sid, "O_warn", r)] and without_news(p[(sid, "O_warn", r)]) == p[(sid, "O", r)]
        assert warn in p[(sid, "R_warn", r)] and without_news(p[(sid, "R_warn", r)]) == p[(sid, "R", r)]
    # O_reveal is O until A sees B's round t0 + REVEAL_AT message, in round t0 + REVEAL_AT + 1
    for r in range(t0 + 1, t0 + REVEAL_AT + 1):
        assert p[(sid, "O_reveal", r)] == p[(sid, "O", r)] and reveal not in p[(sid, "O_reveal", r)]
    for r in range(t0 + REVEAL_AT + 1, t0 + POST_ROUNDS + 1):
        assert reveal in p[(sid, "O_reveal", r)] and "NEWS" not in p[(sid, "O_reveal", r)]
    assert all(reveal not in p[(sid, v, t0 + POST_ROUNDS)] for v in ("O", "O_warn", "R_warn"))


def test_reveal_replaces_the_offer_from_its_round():
    t0, spec = FORK_MIN, SessionSpec("plain_s1", "plain", 1)
    for repeat in (False, True):
        b = ScriptedRival(make_run(spec, DTEXTS, 100, 10), DTEXTS, 1, 0.02, repeat)
        sent = {r: b.move(r, 1.5, VERSIONS["O_reveal"], t0)["message_to_competitor"]
                for r in range(t0 + 1, t0 + POST_ROUNDS + 1)}
        reveal, offer = DTEXTS["reveal"][0], DTEXTS["offers"][1]
        assert [r for r, m in sent.items() if m == reveal] == \
            (list(range(t0 + REVEAL_AT, t0 + POST_ROUNDS + 1)) if repeat else [t0 + REVEAL_AT])
        assert [r for r, m in sent.items() if m == offer] == \
            (list(range(t0 + 1, t0 + REVEAL_AT)) if repeat else [t0 + 1])


def test_disruption_versions_need_their_texts(tmp_path):
    spec = [SessionSpec("plain_s1", "plain", 1)]
    for version in ("O_warn", "R_warn", "O_reveal"):
        with pytest.raises(AssertionError):
            run_forks(ConstantBackend(1.5), spec, TEXTS, 100, tmp_path / version, log=quiet, versions=[version])


def test_forks_from_branches_from_saved_warmups_only(tmp_path):
    src, new = tmp_path / "src", tmp_path / "new"
    first = run_forks(FakeBackend(), [SessionSpec(f"plain_s{k}", "plain", k) for k in (1, 2)], TEXTS, 100, src,
                      log=quiet, versions=["O"])
    backend = ConstantBackend(1.5)
    sessions = [SessionSpec(f"plain_s{k}", "plain", k) for k in (1, 2, 3)]  # s3 has no saved fork: skipped
    second = run_forks(backend, sessions, DTEXTS, 100, new, log=quiet, versions=["O", "O_warn"], forks_from=src)
    assert backend.calls == POST_ROUNDS  # no warm-up was played
    assert sorted((r["session_id"], r["version"]) for r in second) == \
        [(s, v) for s in ("plain_s1", "plain_s2") for v in ("O", "O_warn")]
    old = {r["session_id"]: r for r in first}
    for r in second:
        t0 = old[r["session_id"]]["t0"]
        assert r["t0"] == t0 and r["a_prices"][:t0] == old[r["session_id"]]["a_prices"][:t0]
        assert r["pre_fork_median"] == old[r["session_id"]]["pre_fork_median"]
    cfg = json.loads((new / "config.json").read_text(encoding="utf-8"))
    assert cfg["forks_from"] == str(src) and [s["session_id"] for s in cfg["sessions"]] == ["plain_s1", "plain_s2"]
