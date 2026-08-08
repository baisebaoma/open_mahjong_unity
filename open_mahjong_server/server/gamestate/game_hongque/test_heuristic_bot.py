from .heuristic_bot import (
    V2Value,
    _claim_advances,
    _is_better_ready,
    choose_claim_plan,
    choose_discard,
    choose_turn_plan,
    evaluate_hand,
)
from .rules import call_candidates


def test_heuristic_discard_tenpai_discards_isolated() -> None:
    hand = "AX1 AX2 AX3 BX4 BX5 BX6 CX7 CX8 GY9".split()
    tile, value = choose_discard(hand, [], hand, drawn_tile="GY9")
    assert tile == "GY9"
    assert value.distance == 0
    assert value.ukeire > 0


def test_heuristic_turn_wins_when_winning() -> None:
    plan = choose_turn_plan(
        ["AX1", "AX2", "AX3"],
        [],
        ["AX1", "AX2", "AX3"],
        [],
        supplements=0,
        wall_count=50,
        drawn_tile="AX3",
    )
    assert plan == {"action": "win"}


def test_heuristic_turn_supplements_on_scattered_garbage_draw() -> None:
    hand = "AX1 AX2 AX4 BX5 BX7 CX3 CX8 DX2 DX6 EX1 EX9 FX4 GY9".split()
    plan = choose_turn_plan(
        hand,
        [],
        hand,
        [],
        supplements=0,
        wall_count=50,
        drawn_tile="GY9",
    )
    assert plan == {"action": "supplement"}


def test_heuristic_claim_always_accepts_authoritative_win() -> None:
    plan = choose_claim_plan(
        ["AX1", "AX2"],
        [],
        [{"id": "ron", "kind": "win", "priority": 4}],
        ["AX1", "AX2", "AX3"],
    )
    assert plan == {"action": "claim", "candidate_id": "ron"}


def test_heuristic_claim_accepts_advancing_triplet() -> None:
    hand = "FX1 GY5 DX4 FX3 DX1 BX3 BX4 BY4 AX7 AY6 AY8".split()
    candidates = call_candidates(hand, "EX1")
    plan = choose_claim_plan(hand, [], candidates, hand + ["EX1"])
    assert plan["action"] == "claim"
    chosen = next(item for item in candidates if item["id"] == plan["candidate_id"])
    assert chosen["kind"] == "triplet"
    assert set(chosen["hand_tiles"]) == {"DX1", "FX1"}


def test_heuristic_evaluate_tenpai_reports_points() -> None:
    hand = "AX1 AX2 AX3 BX4 BX5 BX6 CX7 CX8".split()
    value = evaluate_hand(hand, [], hand)
    assert value.distance == 0
    assert value.ukeire > 0
    assert value.points > 0


def test_heuristic_ready_prefers_more_waits_within_point_slack() -> None:
    more_waits = V2Value(0, 3, 80, 0)
    fewer_waits = V2Value(0, 2, 100, 0)
    assert _is_better_ready(more_waits, fewer_waits)


def test_heuristic_ready_trades_waits_only_for_big_points() -> None:
    fewer_waits_big = V2Value(0, 2, 200, 0)
    more_waits_small = V2Value(0, 3, 80, 0)
    assert _is_better_ready(fewer_waits_big, more_waits_small)
    fewer_waits_small = V2Value(0, 2, 100, 0)
    assert not _is_better_ready(fewer_waits_small, more_waits_small)


def test_heuristic_ready_tie_breaks_on_points_then_style() -> None:
    higher_points = V2Value(0, 2, 150, 0)
    lower_points = V2Value(0, 2, 100, 0)
    assert _is_better_ready(higher_points, lower_points)
    assert not _is_better_ready(lower_points, higher_points)


def test_heuristic_claim_refuses_non_advancing_open() -> None:
    before = V2Value(2, 5, 0, 30)
    after = V2Value(2, 5, 0, 20)
    assert not _claim_advances(before, after, priority=1)


def test_heuristic_claim_accepts_rainbow_style_even() -> None:
    before = V2Value(2, 5, 0, 20)
    after = V2Value(2, 5, 0, 20)
    assert _claim_advances(before, after, priority=3)
