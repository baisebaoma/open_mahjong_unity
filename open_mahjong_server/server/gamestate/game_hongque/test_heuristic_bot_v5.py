from .heuristic_bot import choose_discard as choose_discard_v2
from .heuristic_bot_v5 import (
    _threat_level,
    choose_discard,
    choose_turn_plan,
    choose_claim_plan,
)
from .heuristic_bot_v3 import (
    OpponentView,
    _threat_level as _threat_level_v3,
    _tile_danger,
)


def test_v5_discard_tenpai_discards_isolated() -> None:
    hand = "AX1 AX2 AX3 BX4 BX5 BX6 CX7 CX8 GY9".split()
    tile, value = choose_discard(hand, [], hand, drawn_tile="GY9", wall_count=81)
    assert tile == "GY9"
    assert value.distance == 0
    assert value.ukeire > 0


def test_v5_matches_v2_without_opponents() -> None:
    hand = "AX1 AX2 AX3 BX4 BX5 BX6 CX7 CX8 GY9".split()
    t2, _ = choose_discard_v2(hand, [], hand, drawn_tile="GY9")
    t5, _ = choose_discard(hand, [], hand, drawn_tile="GY9", wall_count=81)
    assert t5 == t2


def test_v5_meld_weighted_threat() -> None:
    """Open melds raise the threat level even with short rivers."""
    no_meld = (
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ABC" for i in range(1, 8)))
        for _ in range(3)
    )
    with_meld = (
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ABC" for i in range(1, 8)),
                     meld_tiles=("AX1", "AX2", "AX3", "BX4", "BX5", "BX6"))
        for _ in range(3)
    )
    base = _threat_level_v3(tuple(no_meld), 60)
    enhanced = _threat_level(tuple(with_meld), 60)
    assert base < 1.0  # mid-game so neither sits at the cap
    assert enhanced > base
    assert enhanced <= 1.0


def test_v5_short_rivers_still_no_threat() -> None:
    short = (OpponentView(discards=("AX1", "BX2")) for _ in range(3))
    assert _threat_level(tuple(short), 10) == 0.0


def test_v5_turn_wins_when_winning() -> None:
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


def test_v5_claim_accepts_win() -> None:
    plan = choose_claim_plan(
        ["AX1", "AX2"],
        [],
        [{"id": "ron", "kind": "win", "priority": 4}],
        ["AX1", "AX2", "AX3"],
    )
    assert plan == {"action": "claim", "candidate_id": "ron"}


def test_v5_claim_suppressed_under_threat_with_dangerous_discard() -> None:
    """D3: under threat, a claim forcing a dangerous discard is passed."""
    # The chi (AX2+AX4 with AX3) advances the hand, but every tile it would
    # then shed is in a family the opponents are melding (BX3-5, GX3-5) and
    # not yet discarded (genbutsu), so even the safest discard stays dangerous.
    opponents = (
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ACDEF" for i in range(1, 8)),
                     meld_tiles=("BX3", "BX4", "BX5", "GX3", "GX4", "GX5")),
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ACDEF" for i in range(1, 8)),
                     meld_tiles=("BX3", "BX4", "BX5", "GX3", "GX4", "GX5")),
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ACDEF" for i in range(1, 8)),
                     meld_tiles=("BX3", "BX4", "BX5", "GX3", "GX4", "GX5")),
    )
    hand = [
        "AX2", "AX4",                     # chi pair
        "BX1", "BX2", "BX3",              # completed B group
        "BX7", "BX8",                     # B pair
        "GX1", "GX2", "GX3",              # completed G group
        "GX5", "GX6", "GX7",              # completed G group
    ]
    plan = choose_claim_plan(
        hand, [], [{"id": "chi", "kind": "sequence", "priority": 1, "hand_tiles": ["AX2", "AX4"], "tiles": ["AX3"]}],
        hand, opponents=opponents, wall_count=20,
    )
    # The claim would advance the hand but force a discard near B/G melds; pass.
    assert plan == {"action": "pass"}


def test_v5_claim_normal_without_threat() -> None:
    """Without opponents, claims behave like v3 (accept a good chi)."""
    hand = ["AX2", "AX4", "BX1", "BX3", "CX1", "CX3", "DX1", "DX3", "EX1", "EX3", "FX1", "FX3", "GY9"]
    plan = choose_claim_plan(
        hand, [], [{"id": "chi", "kind": "sequence", "priority": 1, "hand_tiles": ["AX2", "AX4"], "tiles": ["AX3"]}],
        hand,
    )
    assert plan["action"] == "claim"


def test_v5_genbutsu_safe_via_v3_danger() -> None:
    opp = OpponentView(discards=("AX1",), meld_tiles=("BX4", "BX5", "BX6"))
    assert _tile_danger("AX1", (opp,)) == 0.0
