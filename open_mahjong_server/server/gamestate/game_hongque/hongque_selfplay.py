"""Hongque heuristic self-play harness.

Runs 4-seat matches (1 new bot + 3 old efficiency bots, or 4-seat same-policy)
and reports per-seat net points, average rank, win/draw counts, avg fan.

Usage:
    python -m server.gamestate.game_hongque.hongque_selfplay --matches 20 --game-round 4 --new 1
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


def rank_scores(scores: list) -> list:
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    ranks = [0] * 4
    for pos, seat in enumerate(order):
        ranks[seat] = pos + 1
    return ranks


async def run_matches(matches: int, game_round: int, policies: list, base_seed: int = 72001,
                      progress: bool = True, rotate_seat: bool = True) -> dict:
    t0 = time.perf_counter()
    # Stats are tracked per POLICY (index), not per seat — seats rotate.
    per_policy_net = [0.0] * len(policies)
    per_policy_rank = [[] for _ in range(len(policies))]
    per_policy_wins = [0] * len(policies)
    draws = 0
    all_fans: list[int] = []
    for i in range(matches):
        seed = base_seed + i
        # Rotate policies through the 4 seats to cancel dealer / turn-order bias.
        seat_policies = list(policies)
        if rotate_seat:
            shift = i % 4
            seat_policies = [policies[(s - shift) % 4] for s in range(4)]
        result = await play_game(seed, game_round, seat_policies)
        scores = result["scores"]
        ranks = rank_scores(scores)
        # Map per-seat outcome back to the policy that sat there.
        for s in range(4):
            policy_idx = (s - (i % 4)) % 4 if rotate_seat else s
            others = [scores[j] for j in range(4) if j != s]
            per_policy_net[policy_idx] += scores[s] - mean(others)
            per_policy_rank[policy_idx].append(ranks[s])
            per_policy_wins[policy_idx] += result["wins"][s]
        draws += result["draws"]
        all_fans.extend(result["fan_totals"])
        if progress:
            print(f"[{i+1}/{matches}] seed={seed} scores={scores} ranks={ranks}", flush=True)
    elapsed = time.perf_counter() - t0
    out = {
        "matches": matches,
        "elapsed": elapsed,
        "per_seat_net": [round(n, 2) for n in per_policy_net],
        "per_seat_avg_rank": [round(mean(r), 3) for r in per_policy_rank],
        "per_seat_wins": per_policy_wins,
        "draws": draws,
        "avg_fan": round(mean(all_fans), 2) if all_fans else 0,
        "wins_total": sum(per_policy_wins),
        "hands_total": matches * game_round * 4,
    }
    return out


def _seat(name: str):
    if name == "efficiency":
        return EfficiencySeat()
    if name == "heuristic":
        return HeuristicSeat()
    if name == "heuristic-v3":
        return HeuristicV3Seat()
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--game-round", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=72001)
    parser.add_argument("--new", type=str, default="efficiency", help="new seat policy name")
    parser.add_argument("--opponent", type=str, default="efficiency",
                        help="policy name for the other three seats")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-rotate", action="store_true")
    args = parser.parse_args()

    policies = [_seat(args.new)] + [_seat(args.opponent)] * 3
    names = [p.name for p in policies]
    result = asyncio.run(run_matches(
        args.matches, args.game_round, policies, base_seed=args.base_seed,
        progress=not args.quiet, rotate_seat=not args.no_rotate,
    ))
    print(f"\n=== seats={names} ===")
    print(f"matches={result['matches']} elapsed={result['elapsed']:.1f}s "
          f"hands={result['hands_total']}")
    for s in range(4):
        print(f"  seat{s} ({names[s]}): net={result['per_seat_net'][s]:+.1f} "
              f"avg_rank={result['per_seat_avg_rank'][s]} wins={result['per_seat_wins'][s]}")
    print(f"draws={result['draws']} avg_fan={result['avg_fan']}")


if __name__ == "__main__":
    main()
