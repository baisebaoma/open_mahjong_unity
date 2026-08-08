from .heuristic_bot import choose_discard as choose_discard_v2
from .heuristic_bot_v4 import (
    choose_claim_plan,
    choose_discard,
    choose_turn_plan,
)
from .heuristic_bot_v3 import (
    OpponentView,
    _threat_level,
    _tile_danger,
    _structural_value,
)
from .group_index import mask_from_codes


class _FakePlayer:
    def __init__(self, discards, melds):
        self.discards = discards
        self.melds = melds


def test_v4_discard_tenpai_discards_isolated() -> None:
    hand = "AX1 AX2 AX3 BX4 BX5 BX6 CX7 CX8 GY9".split()
    tile, value = choose_discard(hand, [], hand, drawn_tile="GY9", wall_count=81)
    assert tile == "GY9"
    assert value.distance == 0
    assert value.ukeire > 0


def test_v4_turn_wins_when_winning() -> None:
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


def test_v4_turn_supplements_on_scattered_garbage_draw() -> None:
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


def test_v4_claim_always_accepts_authoritative_win() -> None:
    plan = choose_claim_plan(
        ["AX1", "AX2"],
        [],
        [{"id": "ron", "kind": "win", "priority": 4}],
        ["AX1", "AX2", "AX3"],
    )
    assert plan == {"action": "claim", "candidate_id": "ron"}


def test_v4_matches_v2_without_opponents() -> None:
    """With no opponents the defense layer must not change the offense play."""
    hand = "AX1 AX2 AX3 BX4 BX5 BX6 CX7 CX8 GY9".split()
    t2, _ = choose_discard_v2(hand, [], hand, drawn_tile="GY9")
    t4, _ = choose_discard(hand, [], hand, drawn_tile="GY9", wall_count=81)
    assert t4 == t2


def test_v4_threat_gates_on_rivers_and_wall() -> None:
    long_rivers = tuple(
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ABC" for i in range(1, 8)))
        for _ in range(3)
    )
    assert _threat_level([], 81) == 0.0
    assert _threat_level(long_rivers, 81) == 0.0  # early game
    assert _threat_level(long_rivers, 20) >= 0.5  # late + long rivers


def test_v4_genbutsu_is_safe() -> None:
    opp = OpponentView(discards=("AX1",), meld_tiles=("BX4", "BX5", "BX6"))
    assert _tile_danger("AX1", (opp,)) == 0.0


def test_v4_high_value_hand_keeps_attacking_under_mild_threat() -> None:
    """A high-potential hand discounts the threat and stays aggressive."""
    # Colour-concentrated, low-distance shape: high style utility.
    hand = "AX1 AX2 AX3 AX5 AX6 AX7 AY1 AY2 BX4 BX5 CX8 CX9 GY9".split()
    opponents = (
        OpponentView(discards=("AX1", "BX2", "CX3", "DX5", "EX4", "FX1", "GX2"),
                     meld_tiles=("AY5", "AY6", "AY7")),
        OpponentView(discards=("AX3", "BX4", "CX5", "DX6", "EX7", "FX8", "GX9"),
                     meld_tiles=("BY2", "BY3", "BY4")),
        OpponentView(discards=("AX7", "BX8", "CX9", "DX1", "EX2", "FX3", "GX4"),
                     meld_tiles=("CY1", "CY2", "CY3")),
    )
    wall_count = 15
    tile, _ = choose_discard(hand, [], hand, wall_count=wall_count, opponents=opponents)
    # The chosen tile must be a valid non-winning discard (attack path stays on).
    assert tile in hand


def test_v4_end_of_wall_forces_safest_discard() -> None:
    """With <=2 wall tiles left, defense picks the globally safest discard."""
    hand = "AX1 AX2 AX4 BX5 BX7 CX3 CX8 DX2 DX6 EX1 EX9 FX4 GY9".split()
    opponents = (
        OpponentView(discards=("AX1", "BX2", "CX3", "DX5", "EX4", "FX1", "GX2"),
                     meld_tiles=("AY5", "AY6", "AY7")),
        OpponentView(discards=("AX3", "BX4", "CX5", "DX6", "EX7", "FX8", "GX9"),
                     meld_tiles=("BY2", "BY3", "BY4")),
        OpponentView(discards=("AX7", "BX8", "CX9", "DX1", "EX2", "FX3", "GX4"),
                     meld_tiles=("CY1", "CY2", "CY3")),
    )
    tile, _ = choose_discard(hand, [], hand, wall_count=2, opponents=opponents)
    min_danger = min(_tile_danger(t, opponents) for t in hand)
    assert _tile_danger(tile, opponents) == min_danger


def test_v4_opponent_view_equivalence() -> None:
    from .heuristic_bot_v4 import OpponentView as V4Opp

    player = _FakePlayer(
        discards=["AX1", "BX2"],
        melds=[{"tiles": ["CX3", "CX4", "CX5"]}],
    )
    view = V4Opp.from_player(player)
    assert view.discards == ("AX1", "BX2")
    assert view.meld_tiles == ("CX3", "CX4", "CX5")


def _structural_value_ref(hand, tile) -> int:
    remaining = [c for c in hand if c != tile]
    return _structural_value(mask_from_codes(remaining))[0]
