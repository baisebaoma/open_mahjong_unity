"""TEMP: parallel 1v3 bench with per-policy rank series + 95% CI (any policies).

Run from open_mahjong_server:
    uv run --no-project --with fastapi --with scipy python -m \
        server.gamestate.game_hongque._bench_temp \
        --matches 200 --base-seed 73001 --new heuristic-v5 --opponent heuristic-v3
"""
from __future__ import annotations

import argparse
from statistics import mean

import scipy.stats as st  # noqa: E402

from server.gamestate.game_hongque.hongque_selfplay import run_matches_parallel


def report(name: str, ranks: list[int]) -> None:
    n = len(ranks)
    m = mean(ranks)
    se = st.sem(ranks)
    ci = st.t.interval(0.95, n - 1, loc=m, scale=se)
    one_sided_p = st.t.cdf((m - 2.5) / se, n - 1)
    print(f"{name}: n={n} mean={m:.4f} 95%CI=({ci[0]:.4f},{ci[1]:.4f}) "
          f"p(mean<2.5 one-sided)={one_sided_p:.4f}")


def report_net(name: str, nets: list[float]) -> None:
    """Per-match net points (own score - mean of 3 opponents) with 95% CI.

    Each match is one big game = 16 small hands.  A positive lower bound means
    the policy is significantly ahead of its opponents in raw scoring.
    """
    n = len(nets)
    m = mean(nets)
    se = st.sem(nets)
    ci = st.t.interval(0.95, n - 1, loc=m, scale=se)
    per_hand = m / 16.0
    ci_low_hand = ci[0] / 16.0
    one_sided_p = st.t.cdf(-m / se, n - 1) if se > 0 else 1.0
    print(f"{name} net: per-match={m:+.2f} 95%CI=({ci[0]:+.2f},{ci[1]:+.2f}) "
          f"per-hand={per_hand:+.3f} CI_low(per-hand)={ci_low_hand:+.3f} "
          f"p(net>0 one-sided)={one_sided_p:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=72021)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--new", type=str, default="heuristic-v5")
    parser.add_argument("--opponent", type=str, default="heuristic-v3")
    args = parser.parse_args()

    from server.gamestate.game_hongque.hongque_selfplay import _seat
    policies = [_seat(args.new)] + [_seat(args.opponent)] * 3
    result = run_matches_parallel(
        args.matches, 4, policies, base_seed=args.base_seed, workers=args.workers,
        progress=True,
    )
    names = [p.name for p in policies]
    ranks = result["per_policy_rank"]
    print(f"\n=== seats={names} matches={result['matches']} "
          f"elapsed={result['elapsed']:.1f}s hands={result['hands_total']} ===")
    for s in range(4):
        print(f"  seat{s} ({names[s]}): net={result['per_seat_net'][s]:+.1f} "
              f"avg_rank={result['per_seat_avg_rank'][s]} wins={result['per_seat_wins'][s]}")
    print(f"draws={result['draws']} avg_fan={result['avg_fan']}")
    print()
    nets = result["per_policy_game_net"]
    for s in range(4):
        report(names[s], ranks[s])
        report_net(names[s], nets[s])


if __name__ == "__main__":
    main()
