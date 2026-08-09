from .heuristic_bot_v7 import choose_claim_plan
from .heuristic_bot_v3 import OpponentView


def test_v7_claim_accepts_win() -> None:
    plan = choose_claim_plan(
        ["AX1", "AX2"],
        [],
        [{"id": "ron", "kind": "win", "priority": 4}],
        ["AX1", "AX2", "AX3"],
    )
    assert plan == {"action": "claim", "candidate_id": "ron"}


def test_v7_claim_suppressed_under_threat_with_dangerous_discard() -> None:
    """D3 (isolated): under threat, a claim forcing a dangerous discard is passed."""
    opponents = (
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ACDEF" for i in range(1, 8)),
                     meld_tiles=("BX3", "BX4", "BX5", "GX3", "GX4", "GX5")),
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ACDEF" for i in range(1, 8)),
                     meld_tiles=("BX3", "BX4", "BX5", "GX3", "GX4", "GX5")),
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ACDEF" for i in range(1, 8)),
                     meld_tiles=("BX3", "BX4", "BX5", "GX3", "GX4", "GX5")),
    )
    hand = [
        "AX2", "AX4",
        "BX1", "BX2", "BX3",
        "BX7", "BX8",
        "GX1", "GX2", "GX3",
        "GX5", "GX6", "GX7",
    ]
    plan = choose_claim_plan(
        hand, [], [{"id": "chi", "kind": "sequence", "priority": 1, "hand_tiles": ["AX2", "AX4"], "tiles": ["AX3"]}],
        hand, opponents=opponents, wall_count=20,
    )
    assert plan == {"action": "pass"}


def test_v7_claim_normal_without_threat() -> None:
    """Without opponents, claims behave like v3 (accept a good chi)."""
    hand = ["AX2", "AX4", "BX1", "BX3", "CX1", "CX3", "DX1", "DX3", "EX1", "EX3", "FX1", "FX3", "GY9"]
    plan = choose_claim_plan(
        hand, [], [{"id": "chi", "kind": "sequence", "priority": 1, "hand_tiles": ["AX2", "AX4"], "tiles": ["AX3"]}],
        hand,
    )
    assert plan["action"] == "claim"
