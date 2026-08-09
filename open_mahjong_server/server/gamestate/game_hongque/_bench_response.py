"""TEMP: 对比 efficiency（牌效罗伯特）vs heuristic-v3 的单步决策耗时。

串行自战若干场，包装 seat 记录每次 turn/claim 的纯计算耗时，输出
mean/median/p95/max。用于 PR 验证虹雀 v3 响应不比牌效罗伯特慢多少。
"""
from __future__ import annotations

import argparse
import asyncio
import time
from statistics import mean, median

from server.gamestate.game_hongque.hongque_selfplay import _seat, play_game


class TimingSeat:
    """Wrap a real seat and time each decision."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.turns: list[float] = []
        self.claims: list[float] = []

    def turn(self, state, player):
        t0 = time.perf_counter()
        result = self.inner.turn(state, player)
        self.turns.append(time.perf_counter() - t0)
        return result

    def claim(self, state, player, candidates):
        t0 = time.perf_counter()
        result = self.inner.claim(state, player, candidates)
        self.claims.append(time.perf_counter() - t0)
        return result


def fmt(label: str, dts: list[float]) -> None:
    if not dts:
        print(f"{label}: no samples")
        return
    ordered = sorted(dts)
    n = len(ordered)
    print(
        f"{label}: n={n} mean={mean(ordered)*1000:.2f}ms "
        f"median={median(ordered)*1000:.2f}ms "
        f"p95={ordered[int(n*0.95)-1]*1000:.2f}ms "
        f"p99={ordered[int(n*0.99)-1]*1000:.2f}ms "
        f"max={ordered[-1]*1000:.2f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=72021)
    args = parser.parse_args()

    eff = TimingSeat(_seat("efficiency"))
    v3 = TimingSeat(_seat("heuristic-v3"))
    # 2 v3 + 2 efficiency 同场，轮换座位，双方样本量接近。
    policies = [v3, v3, eff, eff]
    t0 = time.perf_counter()
    for i in range(args.matches):
        seed = args.base_seed + i
        shift = i % 4  # 轮换座位，与自战基准同构
        seat_policies = [policies[(s - shift) % 4] for s in range(4)]
        asyncio.run(play_game(seed, 4, seat_policies))
    print(f"elapsed={time.perf_counter()-t0:.1f}s matches={args.matches}")
    print()
    fmt("efficiency.turn", eff.turns)
    fmt("efficiency.claim", eff.claims)
    fmt("v3.turn", v3.turns)
    fmt("v3.claim", v3.claims)
    print()
    if eff.turns and v3.turns:
        ratio = mean(v3.turns) / mean(eff.turns)
        print(f"v3.turn / efficiency.turn mean ratio = {ratio:.2f}x")
    if eff.claims and v3.claims:
        ratio = mean(v3.claims) / mean(eff.claims)
        print(f"v3.claim / efficiency.claim mean ratio = {ratio:.2f}x")


if __name__ == "__main__":
    main()
