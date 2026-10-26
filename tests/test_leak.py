"""The summary versions hide B's price line, but A's own quantity and profit still determine it."""

import re

from pilot.analyse_leak import env, grid_audit, recover_from_profit, recover_from_quantity, shown


def test_rival_price_recovered_from_own_numbers():
    e = env()
    for p_a in (1.50, 1.62, 1.75):
        for p_b in (1.47, 1.65, 1.90):
            q, pi = shown(e, p_a, p_b)
            assert abs(recover_from_quantity(p_a, q) - p_b) < 0.005
            assert abs(recover_from_profit(p_a, pi) - p_b) < 0.005


def test_whole_grid_within_one_cent():
    g = grid_audit(env())
    assert g["share_within_0.01_from_quantity"] == 1.0
    assert g["share_within_0.01_from_profit"] == 1.0


def test_fork_regex_keeps_quantity_and_profit():
    block = "- My price: 1.62\n- Competitor's prices: {'Firm B': 1.9}\n- My quantity sold: 64.73\n- My profit earned: 40.13\n"
    stripped = re.sub(r"- Competitor's prices:[^\n]*\n", "", block)  # as in pilot/fork.py (history_mode="summary")
    assert "Competitor" not in stripped
    assert "My quantity sold" in stripped and "My profit earned" in stripped
