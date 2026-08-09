"""Hongque heuristic self-play harness.

Runs 4-seat matches (1 new bot + 3 old efficiency bots, or 4-seat same-policy)
and reports per-seat net points, average rank, win/draw counts, avg fan.

Parallel mode (default): seeds are sharded across ``--workers`` processes
(``ProcessPoolExecutor``), each worker runs a contiguous seed range in its own
event loop, results are merged in seed order.  This saturates all cores; use
``--workers 1`` for the serial path.

Usage:
    python -m server.gamestate.game_hongque.hongque_selfplay --matches 100 --new heuristic-v3 --opponent efficiency
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from server.gamestate.game_hongque.HongqueGameState import HongqueGameState  # noqa: E402
from server.gamestate.game_hongque.rules import kong_candidates  # noqa: E402
from server.gamestate.game_hongque.scoring import best_win_result  # noqa: E402

USER_IDS = [101, 102, 103, 104]


def build_room(seed: int, game_round: int) -> dict:
    return {
        "room_id": f"sp-{seed}",
        "room_type": "custom",
        "room_rule": "hongque",
        "sub_rule": "hongque/v1.6",
        "game_round": game_round,
        "random_seed": seed,
        "tips": False,
        "round_timer": 20,
        "step_timer": 5,
        "tactical_pre_grace_delay": 0.0,
        "tactical_grace_seconds": 0.0,
        "player_list": list(USER_IDS),
        "player_settings": {},
    }


def visible_codes(state, player_index: int) -> list:
    return list(state._visible_codes_for(player_index))


async def play_game(seed: int, game_round: int, policies: list, max_ticks: int = 300000) -> dict:
    room = build_room(seed, game_round)
    state = HongqueGameState(None, room, gamestate_id=f"sp-{seed}")

    guard = 0
    win_count = [0, 0, 0, 0]
    draw_count = 0
    fan_totals: list[int] = []
    await state._start_round()
    max_round = state.max_round
    while state.phase != "game_end":
        guard += 1
        if guard > max_ticks:
            raise RuntimeError(f"stuck: round={state.current_round} phase={state.phase} wall={len(state.wall)} tick={state.action_tick}")

        if state.phase == "turn":
            idx = state.current_player_index
            player = state.players[idx]
            plan = policies[idx].turn(state, player)
            action = plan["action"]
            if action == "win":
                await state._handle_turn_action(player, "win", None, None)
                _count_result(state, win_count, fan_totals)
            elif action == "supplement":
                await state._handle_turn_action(player, "supplement", None, None)
            elif action == "kong":
                await state._handle_turn_action(player, "kong", None, plan.get("candidate_id"))
            elif action == "discard":
                await state._discard_and_open_claim(player, plan["tile"])
            else:
                raise RuntimeError(f"unknown turn plan {plan}")
            _cancel_tasks(state)
        elif state.phase == "claim":
            ron_claimed = False
            for pid, opts in list(state.claim_options.items()):
                if pid in state.claim_responses:
                    continue
                player = state.players[pid]
                plan = policies[pid].claim(state, player, opts)
                if plan["action"] == "claim":
                    cand = next(c for c in opts if c["id"] == plan["candidate_id"])
                    state.claim_responses[pid] = {"action": "claim", "candidate": cand}
                    if cand.get("kind") == "win":
                        ron_claimed = True
                else:
                    state.claim_responses[pid] = {"action": "pass"}
            await state._resolve_claims()
            if ron_claimed and state.phase == "round_end":
                _count_result(state, win_count, fan_totals)
            _cancel_tasks(state)
        elif state.phase == "round_end":
            if state.round_result and state.round_result.get("reason") == "draw":
                draw_count += 1
            if state._round_task and not state._round_task.done():
                state._round_task.cancel()
            state.current_round += 1
            if state.current_round > max_round:
                state.phase = "game_end"
                break
            await state._start_round()
            _cancel_tasks(state)
        else:
            raise RuntimeError(f"unexpected phase {state.phase}")

    return {
        "scores": [p.score for p in state.players],
        "round": state.current_round,
        "ticks": guard,
        "wins": win_count,
        "draws": draw_count,
        "fan_totals": fan_totals,
    }


def _count_result(state, win_count: list, fan_totals: list) -> None:
    if state.round_result is None:
        return
    for w in state.round_result.get("winners", ()):
        win_count[w["player"]] += 1
        fan_totals.append(w.get("fan_total", 0))


def _cancel_tasks(state) -> None:
    for attr in ("_bot_task", "_claim_timeout_task", "_turn_timeout_task",
                 "_claim_grace_task", "_claim_grace_timeout_task", "_round_task"):
        task = getattr(state, attr, None)
        if isinstance(task, dict):
            for t in list(task.values()):
                if t and not t.done():
                    t.cancel()
        elif task and not task.done():
            task.cancel()


# ── Seats ────────────────────────────────────────────────────────────────────

class EfficiencySeat:
    """Mirror of the upstream user_id==2 efficiency bot."""
    name = "efficiency"

    def turn(self, state, player):
        from server.gamestate.game_hongque.efficiency_bot import choose_turn_plan
        return choose_turn_plan(
            player.hand, player.melds, visible_codes(state, player.index),
            kong_candidates(player.hand, player.melds),
            supplements=player.supplements, wall_count=len(state.wall),
            drawn_tile=player.drawn_tile, last_draw_was_supplement=player.last_draw_was_supplement,
        )

    def claim(self, state, player, candidates):
        from server.gamestate.game_hongque.efficiency_bot import choose_claim_plan
        return choose_claim_plan(player.hand, player.melds, candidates, visible_codes(state, player.index))


class HeuristicSeat:
    """v2 score-aware heuristic bot (user_id=3)."""
    name = "heuristic"

    def turn(self, state, player):
        from server.gamestate.game_hongque.heuristic_bot import choose_turn_plan
        return choose_turn_plan(
            player.hand, player.melds, visible_codes(state, player.index),
            kong_candidates(player.hand, player.melds),
            supplements=player.supplements, wall_count=len(state.wall),
            drawn_tile=player.drawn_tile, last_draw_was_supplement=player.last_draw_was_supplement,
        )

    def claim(self, state, player, candidates):
        from server.gamestate.game_hongque.heuristic_bot import choose_claim_plan
        return choose_claim_plan(player.hand, player.melds, candidates, visible_codes(state, player.index))


class HeuristicV3Seat:
    """v3 heuristic + threat-gated folding (defense) bot."""
    name = "heuristic-v3"

    def _opponents(self, state, player_index):
        from server.gamestate.game_hongque.heuristic_bot_v3 import OpponentView
        return tuple(
            OpponentView.from_player(player)
            for player in state.players
            if player.index != player_index
        )

    def turn(self, state, player):
        from server.gamestate.game_hongque.heuristic_bot_v3 import choose_turn_plan
        return choose_turn_plan(
            player.hand, player.melds, visible_codes(state, player.index),
            kong_candidates(player.hand, player.melds),
            supplements=player.supplements, wall_count=len(state.wall),
            drawn_tile=player.drawn_tile, last_draw_was_supplement=player.last_draw_was_supplement,
            opponents=self._opponents(state, player.index),
        )

    def claim(self, state, player, candidates):
        from server.gamestate.game_hongque.heuristic_bot_v3 import choose_claim_plan
        return choose_claim_plan(player.hand, player.melds, candidates, visible_codes(state, player.index))


class HeuristicV4Seat:
    """v4 heuristic + own-value-aware defense + end-of-wall genbutsu."""
    name = "heuristic-v4"

    def _opponents(self, state, player_index):
        from server.gamestate.game_hongque.heuristic_bot_v3 import OpponentView
        return tuple(
            OpponentView.from_player(player)
            for player in state.players
            if player.index != player_index
        )

    def turn(self, state, player):
        from server.gamestate.game_hongque.heuristic_bot_v4 import choose_turn_plan
        return choose_turn_plan(
            player.hand, player.melds, visible_codes(state, player.index),
            kong_candidates(player.hand, player.melds),
            supplements=player.supplements, wall_count=len(state.wall),
            drawn_tile=player.drawn_tile, last_draw_was_supplement=player.last_draw_was_supplement,
            opponents=self._opponents(state, player.index),
        )

    def claim(self, state, player, candidates):
        from server.gamestate.game_hongque.heuristic_bot_v4 import choose_claim_plan
        return choose_claim_plan(player.hand, player.melds, candidates, visible_codes(state, player.index))


class HeuristicV5Seat:
    """v5 heuristic + meld-weighted threat gate (D1)."""
    name = "heuristic-v5"

    def _opponents(self, state, player_index):
        from server.gamestate.game_hongque.heuristic_bot_v3 import OpponentView
        return tuple(
            OpponentView.from_player(player)
            for player in state.players
            if player.index != player_index
        )

    def turn(self, state, player):
        from server.gamestate.game_hongque.heuristic_bot_v5 import choose_turn_plan
        return choose_turn_plan(
            player.hand, player.melds, visible_codes(state, player.index),
            kong_candidates(player.hand, player.melds),
            supplements=player.supplements, wall_count=len(state.wall),
            drawn_tile=player.drawn_tile, last_draw_was_supplement=player.last_draw_was_supplement,
            opponents=self._opponents(state, player.index),
        )

    def claim(self, state, player, candidates):
        from server.gamestate.game_hongque.heuristic_bot_v5 import choose_claim_plan
        return choose_claim_plan(
            player.hand, player.melds, candidates, visible_codes(state, player.index),
            opponents=self._opponents(state, player.index), wall_count=len(state.wall),
        )


class HeuristicV6Seat:
    """v6 heuristic + tenpai tie-break on safety (E3), v3 threat/claims."""
    name = "heuristic-v6"

    def _opponents(self, state, player_index):
        from server.gamestate.game_hongque.heuristic_bot_v3 import OpponentView
        return tuple(
            OpponentView.from_player(player)
            for player in state.players
            if player.index != player_index
        )

    def turn(self, state, player):
        from server.gamestate.game_hongque.heuristic_bot_v6 import choose_turn_plan
        return choose_turn_plan(
            player.hand, player.melds, visible_codes(state, player.index),
            kong_candidates(player.hand, player.melds),
            supplements=player.supplements, wall_count=len(state.wall),
            drawn_tile=player.drawn_tile, last_draw_was_supplement=player.last_draw_was_supplement,
            opponents=self._opponents(state, player.index),
        )

    def claim(self, state, player, candidates):
        from server.gamestate.game_hongque.heuristic_bot_v6 import choose_claim_plan
        return choose_claim_plan(
            player.hand, player.melds, candidates, visible_codes(state, player.index),
            opponents=self._opponents(state, player.index), wall_count=len(state.wall),
        )


class HeuristicV7Seat:
    """v7 heuristic + D3 claim suppression only (v3 threat)."""
    name = "heuristic-v7"

    def _opponents(self, state, player_index):
        from server.gamestate.game_hongque.heuristic_bot_v3 import OpponentView
        return tuple(
            OpponentView.from_player(player)
            for player in state.players
            if player.index != player_index
        )

    def turn(self, state, player):
        from server.gamestate.game_hongque.heuristic_bot_v7 import choose_turn_plan
        return choose_turn_plan(
            player.hand, player.melds, visible_codes(state, player.index),
            kong_candidates(player.hand, player.melds),
            supplements=player.supplements, wall_count=len(state.wall),
            drawn_tile=player.drawn_tile, last_draw_was_supplement=player.last_draw_was_supplement,
            opponents=self._opponents(state, player.index),
        )

    def claim(self, state, player, candidates):
        from server.gamestate.game_hongque.heuristic_bot_v7 import choose_claim_plan
        return choose_claim_plan(
            player.hand, player.melds, candidates, visible_codes(state, player.index),
            opponents=self._opponents(state, player.index), wall_count=len(state.wall),
        )


def rank_scores(scores: list) -> list:
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    ranks = [0] * 4
    for pos, seat in enumerate(order):
        ranks[seat] = pos + 1
    return ranks


def _run_seed_range(base_seed: int, matches: int, game_round: int, policy_names: list,
                    rotate_seat: bool = True, global_base: int = 0) -> dict:
    """ProcessPool worker: run a contiguous seed range; returns per-policy stats.

    ``policy_names`` is a list of 4 seat names (reconstructed via ``_seat`` in
    this worker process, so the payload stays picklable across spawn).
    ``global_base`` is the first seed of the whole batch, so seat rotation uses
    the global match index (not the shard-local one) — this keeps the sharded
    results byte-identical to the serial path.
    """
    policies = [_seat(n) for n in policy_names]
    per_policy_net = [0.0] * len(policies)
    per_policy_rank: list[list[int]] = [[] for _ in range(len(policies))]
    per_policy_wins = [0] * len(policies)
    per_policy_game_net: list[list[float]] = [[] for _ in range(len(policies))]
    draws = 0
    all_fans: list[int] = []
    for i in range(matches):
        seed = base_seed + i
        g = (seed - global_base) if global_base else i  # global match idx
        seat_policies = list(policies)
        if rotate_seat:
            shift = g % 4
            seat_policies = [policies[(s - shift) % 4] for s in range(4)]
        result = asyncio.run(play_game(seed, game_round, seat_policies))
        scores = result["scores"]
        ranks = rank_scores(scores)
        for s in range(4):
            policy_idx = (s - (g % 4)) % 4 if rotate_seat else s
            others = [scores[j] for j in range(4) if j != s]
            net = scores[s] - mean(others)
            per_policy_net[policy_idx] += net
            per_policy_game_net[policy_idx].append(net)
            per_policy_rank[policy_idx].append(ranks[s])
            per_policy_wins[policy_idx] += result["wins"][s]
        draws += result["draws"]
        all_fans.extend(result["fan_totals"])
    return {
        "matches": matches,
        "per_policy_net": per_policy_net,
        "per_policy_game_net": per_policy_game_net,
        "per_policy_rank": per_policy_rank,
        "per_policy_wins": per_policy_wins,
        "draws": draws,
        "fan_totals": all_fans,
    }


def _merge_ranges(partials: list, matches: int, game_round: int) -> dict:
    n_policies = len(partials[0]["per_policy_net"])
    per_policy_net = [sum(p["per_policy_net"][i] for p in partials) for i in range(n_policies)]
    per_policy_rank: list[list[int]] = [[] for _ in range(n_policies)]
    per_policy_game_net: list[list[float]] = [[] for _ in range(n_policies)]
    for p in partials:
        for i in range(n_policies):
            per_policy_rank[i].extend(p["per_policy_rank"][i])
            per_policy_game_net[i].extend(p["per_policy_game_net"][i])
    per_policy_wins = [sum(p["per_policy_wins"][i] for p in partials) for i in range(n_policies)]
    draws = sum(p["draws"] for p in partials)
    all_fans: list[int] = []
    for p in partials:
        all_fans.extend(p["fan_totals"])
    return {
        "matches": matches,
        "elapsed": sum(p.get("elapsed", 0.0) for p in partials),
        "per_seat_net": [round(n, 2) for n in per_policy_net],
        "per_seat_avg_rank": [round(mean(r), 3) if r else 0.0 for r in per_policy_rank],
        "per_policy_rank": per_policy_rank,
        "per_policy_game_net": per_policy_game_net,
        "per_seat_wins": per_policy_wins,
        "draws": draws,
        "avg_fan": round(mean(all_fans), 2) if all_fans else 0,
        "wins_total": sum(per_policy_wins),
        "hands_total": matches * game_round * 4,
    }


async def run_matches(matches: int, game_round: int, policies: list, base_seed: int = 72001,
                      progress: bool = True, rotate_seat: bool = True) -> dict:
    """Serial runner (kept for compatibility); prefer ``run_matches_parallel``."""
    t0 = time.perf_counter()
    partial = _run_seed_range(
        base_seed, matches, game_round,
        [p.name for p in policies], rotate_seat=rotate_seat, global_base=base_seed,
    )
    partial["elapsed"] = time.perf_counter() - t0
    return _merge_ranges([partial], matches, game_round)


def run_matches_parallel(matches: int, game_round: int, policies: list, base_seed: int = 72001,
                         workers: int = 0, progress: bool = True, rotate_seat: bool = True) -> dict:
    """Shard ``matches`` seeds across ``workers`` processes and merge results."""
    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)
    workers = min(workers, matches)
    policy_names = [p.name for p in policies]

    # Contiguous seed shards, one per worker.
    shard_counts = [matches // workers] * workers
    for i in range(matches % workers):
        shard_counts[i] += 1
    cursor = base_seed
    shards: list[tuple] = []
    for count in shard_counts:
        if count > 0:
            shards.append((cursor, count, game_round, policy_names, rotate_seat, base_seed))
            cursor += count

    t0 = time.perf_counter()
    if len(shards) == 1:
        partial = _run_seed_range(*shards[0])
        partial["elapsed"] = time.perf_counter() - t0
        return _merge_ranges([partial], matches, game_round)

    partials = []
    with ProcessPoolExecutor(max_workers=len(shards)) as pool:
        futs = {pool.submit(_run_seed_range, *sh): sh[0] for sh in shards}
        done = 0
        for fut in as_completed(futs):
            done += 1
            partials.append(fut.result())
            if progress:
                print(f"[worker {done}/{len(shards)}] seeds {futs[fut]}+ done", flush=True)
    merged = _merge_ranges(partials, matches, game_round)
    merged["elapsed"] = time.perf_counter() - t0  # wall clock, not worker-sum
    return merged


def _seat(name: str):
    if name == "efficiency":
        return EfficiencySeat()
    if name == "heuristic":
        return HeuristicSeat()
    if name == "heuristic-v3":
        return HeuristicV3Seat()
    if name == "heuristic-v4":
        return HeuristicV4Seat()
    if name == "heuristic-v5":
        return HeuristicV5Seat()
    if name == "heuristic-v6":
        return HeuristicV6Seat()
    if name == "heuristic-v7":
        return HeuristicV7Seat()
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--game-round", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=72001)
    parser.add_argument("--new", type=str, default="efficiency", help="new seat policy name")
    parser.add_argument("--opponent", type=str, default="efficiency",
                        help="policy name for the other three seats")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel processes (0 = auto, all cores)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-rotate", action="store_true")
    args = parser.parse_args()

    policies = [_seat(args.new)] + [_seat(args.opponent)] * 3
    names = [p.name for p in policies]
    result = run_matches_parallel(
        args.matches, args.game_round, policies, base_seed=args.base_seed,
        workers=args.workers, progress=not args.quiet, rotate_seat=not args.no_rotate,
    )
    print(f"\n=== seats={names} ===")
    print(f"matches={result['matches']} elapsed={result['elapsed']:.1f}s "
          f"hands={result['hands_total']}")
    for s in range(4):
        print(f"  seat{s} ({names[s]}): net={result['per_seat_net'][s]:+.1f} "
              f"avg_rank={result['per_seat_avg_rank'][s]} wins={result['per_seat_wins'][s]}")
    print(f"draws={result['draws']} avg_fan={result['avg_fan']}")


if __name__ == "__main__":
    main()
