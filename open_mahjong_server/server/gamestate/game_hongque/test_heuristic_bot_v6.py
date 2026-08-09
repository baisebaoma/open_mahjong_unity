from .heuristic_bot_v6 import (
    _pick_tenpai_discard,
    choose_discard,
)
from .heuristic_bot_v3 import (
    OpponentView,
    choose_discard as choose_discard_v3,
)


def test_v6_tenpai_picks_drawn_when_safe() -> None:
    """E3 keeps the v2/v3 habit of shedding the draw when it is not risky."""
    opp = OpponentView(discards=("AX1", "BX2", "CX3"))
    picked = _pick_tenpai_discard(("AX1", "BX2", "CX3"), drawn_tile="CX3", opponents=(opp,))
    # All three are genbutsu against this single opponent → equally safe.
    assert picked == "CX3"


def test_v6_tenpai_avoids_dangerous_draw() -> None:
    """E3 sheds the drawn tile only when a strictly safer equal-value discard exists."""
    opponents = (
        OpponentView(discards=tuple(f"{c}X{i}" for c in "CDEFG" for i in range(1, 8)),
                     meld_tiles=("BX3", "BX4", "BX5")),
        OpponentView(discards=tuple(f"{c}X{i}" for c in "CDEFG" for i in range(1, 8)),
                     meld_tiles=("BX3", "BX4", "BX5")),
        OpponentView(discards=tuple(f"{c}X{i}" for c in "CDEFG" for i in range(1, 8)),
                     meld_tiles=("BX3", "BX4", "BX5")),
    )
    ties = ("BX1", "CX7", "EX9")
    picked = _pick_tenpai_discard(ties, drawn_tile="BX1", opponents=opponents)
    assert picked == "CX7"  # near the B meld → dangerous; pick a safe tie instead


def test_v6_discard_matches_v3_without_opponents() -> None:
    hand = "AX1 AX2 AX3 BX4 BX5 BX6 CX7 CX8 GY9".split()
    t3, _ = choose_discard_v3(hand, [], hand, drawn_tile="GY9", wall_count=81)
    t6, _ = choose_discard(hand, [], hand, drawn_tile="GY9", wall_count=81)
    assert t6 == t3


def test_v6_discard_tenpai_avoids_risky_draw() -> None:
    """E3 in choose_discard: a dangerous drawn tile is replaced by a safe tie."""
    opponents = (
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ABDEF" for i in range(1, 8)),
                     meld_tiles=("CX3", "CX4", "CX5")),
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ABDEF" for i in range(1, 8)),
                     meld_tiles=("CX3", "CX4", "CX5")),
        OpponentView(discards=tuple(f"{c}X{i}" for c in "ABDEF" for i in range(1, 8)),
                     meld_tiles=("CX3", "CX4", "CX5")),
    )
    # Real 12-tile hand: discarding CX3 (keeps GX7-8 wait for GX9) or GX8
    # (keeps GX6-7 wait for GX5) are exact ties — identical ukeire (1) and
    # points (65, both complete a 一条龙).  The drawn CX3 sits inside the
    # opponents' C meld (danger 5/opp); GX8 is far from it (danger 0).
    hand = ["AX1", "AX2", "AX3", "BX7", "BX8", "BX9",
            "CX3", "CX4", "CX5", "GX6", "GX7", "GX8"]
    tile, _ = choose_discard(hand, [], hand, drawn_tile="CX3",
                             opponents=opponents, wall_count=81)
    assert tile == "GX8"
    # Without opponents the danger tie-break is inert → v3 behavior: shed draw.
    tile_v3, _ = choose_discard_v3(hand, [], hand, drawn_tile="CX3", wall_count=81)
    assert tile_v3 == "CX3"
