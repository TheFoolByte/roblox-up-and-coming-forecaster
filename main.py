#!/usr/bin/env python3
"""Predict Roblox's Up-and-Coming top 10.

Monte Carlo log-rank diffusion, time-averaged over the scored window,
with momentum, playing-growth drift, a breakout boost for games whose
playing is surging, and a 28-day tenure hazard when the history carries
a game's creation date.

Stdlib only, deterministic (fixed seed), no network.
Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR
(default /output) / forecast.json as named in task.json.
"""
import json
import math
import os
import random
import statistics
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
SEED = 20261112
TRIALS = 1500
STEP_H = 3.0
CAP = 0.25
SUM_TARGET = 0.85
# Tuned (v3 aggressive): Student-t innovations (df 5) match the heavy tails of
# log-rank moves (kurtosis ~15.8); breakout blend at 0.6x — smaller than the
# safe 0.75x, a bet that the final window behaves like the late rolling cuts
# (where 0.6x won biggest), at the cost of ~0.13 practice score.
T_DF = 5.0
BREAKOUT_SCALE = 0.6


def parse_time(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("at", ""))
    return rows


def nearest_before(series, target_ts, field):
    """Latest value of field at or before target_ts in a time-sorted series."""
    best = None
    for t, v in series:
        if t <= target_ts and v is not None:
            best = v
        elif t > target_ts:
            break
    return best


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    out_name = task.get("output", "forecast.json")
    window = task.get("window", {}) or {}
    win_from = parse_time(window.get("from")) or parse_time(task.get("history_until"))
    win_until = parse_time(window.get("until"))
    horizon_h = 72.0
    if win_from and win_until:
        horizon_h = max(1.0, (win_until - win_from).total_seconds() / 3600.0)

    rows = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    if not rows:
        raise SystemExit("empty history")
    last = rows[-1]
    last_t = parse_time(last.get("at"))
    if win_from is None:
        win_from = last_t

    # Per-game series: (timestamp, rank, playing), plus latest details.
    rank_series = {}
    play_series = {}
    visits_series = {}
    details = {}
    for r in rows:
        t = parse_time(r.get("at"))
        if t is None:
            continue
        for g in r.get("up_and_coming") or []:
            uid = g.get("universe_id")
            if uid is None:
                continue
            rank_series.setdefault(uid, []).append((t, g.get("rank")))
            play_series.setdefault(uid, []).append((t, g.get("playing")))
        det = r.get("details") or {}
        if isinstance(det, dict):
            for k, v in det.items():
                if v:
                    try:
                        uid = int(k)
                    except (TypeError, ValueError):
                        continue
                    details[uid] = v
                    if v.get("visits") is not None:
                        visits_series.setdefault(uid, []).append((t, v.get("visits")))

    last_chart = sorted(last.get("up_and_coming") or [],
                        key=lambda g: g.get("rank", 999))
    uids = [g["universe_id"] for g in last_chart if g.get("universe_id") is not None]
    if not uids:
        raise SystemExit("no games in last chart")

    last_ts = last_t.timestamp() if last_t else 0.0

    def value_at(series, uid, hours_back):
        pts = series.get(uid) or []
        target = last_ts - hours_back * 3600.0
        best = None
        for t, v in pts:
            if t.timestamp() <= target and v is not None:
                best = v
            elif t.timestamp() > target:
                break
        return best

    def rank_at(uid, hours_back):
        return value_at(rank_series, uid, hours_back)

    states = []
    breakout = {}
    for g in last_chart:
        uid = g["universe_id"]
        r0 = g.get("rank") or 99
        lr = math.log(max(1, r0))
        pts = rank_series.get(uid) or []
        # Mean log-rank over the last 48h.
        recent = [math.log(r) for (t, r) in pts
                  if r and last_ts - t.timestamp() <= 48 * 3600.0]
        mean_lr = statistics.mean(recent) if recent else lr
        # Velocity: slope of log-rank vs time over the last 24h.
        v24 = 0.0
        win_pts = [(t.timestamp(), math.log(r)) for (t, r) in pts
                   if r and last_ts - t.timestamp() <= 24 * 3600.0]
        if len(win_pts) >= 2:
            xs = [x for x, _ in win_pts]
            ys = [y for _, y in win_pts]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            if den > 0:
                # slope per second -> per hour
                v24 = sum((x - mx) * (y - my) for x, y in win_pts) / den * 3600.0
        v24 = max(-0.02, min(0.02, v24))
        # Playing growth.
        p_now = g.get("playing")
        p24 = value_at(play_series, uid, 24)
        p48 = value_at(play_series, uid, 48)
        # Playing change since the previous snapshot (a surge in one
        # step matters most for games first seen only a few hours ago,
        # where a 6h/24h lookback finds nothing).
        prev_pts = play_series.get(uid) or []
        p_prev = prev_pts[-2][1] if len(prev_pts) >= 2 else None
        g_step = math.log(p_now / p_prev) if p_now and p_prev else 0.0
        g24 = math.log(p_now / p24) if p_now and p24 else 0.0
        g48 = math.log(p_now / p48) if p_now and p48 else 0.0
        play_v = -0.01 * (g24 + g48)
        play_v = max(-0.04, min(0.04, play_v))
        # Visits growth, when details carry it (final history only).
        v24g = 0.0
        if visits_series.get(uid):
            vn = visits_series[uid][-1][1]
            v_old = value_at(visits_series, uid, 24)
            if vn and v_old and vn > v_old:
                v24g = math.log(vn / v_old)
        # Sigma per hour, from the residual spread of log-rank moves:
        # top places move most, deep places least.
        if r0 <= 5:
            sigma = 0.131
        elif r0 <= 10:
            sigma = 0.103
        elif r0 <= 20:
            sigma = 0.086
        elif r0 <= 50:
            sigma = 0.078
        else:
            sigma = 0.055
        first_t = pts[0][0].timestamp() if pts else last_ts
        if last_ts - first_t < 48 * 3600.0:
            sigma *= 1.2
        # Tenure hazard: created + 28 days.
        expiry_ts = None
        det = details.get(uid) or {}
        created = parse_time(det.get("created")) if det else None
        if created is not None:
            expiry_ts = created.timestamp() + 28 * 86400.0
        # Breakout score: surging playing, and a big climb over 72h.
        rank72 = rank_at(uid, 72)
        d72 = (r0 - rank72) if rank72 else 0
        score = 0.0
        if g24 > 0.8:
            score += g24 - 0.8
        if g_step > 0.5:
            score += (g_step - 0.5) * 0.7
        if v24g > 0.8:
            score += (v24g - 0.8) * 0.5
        if d72 < -30:
            score += (-d72 - 30) / 60.0
        # Debut boost: a game first seen < 48h ago whose playing is
        # surging has almost no rank history, so its MC diffusion is
        # anchored on a noisy rank; give its breakout score a flat
        # bonus so the blend allocates it more top-place mass.
        if last_ts - first_t < 48 * 3600.0 and (g_step > 0.5 or g24 > 0.8):
            score += 0.5
        if score > 0:
            breakout[uid] = score
        states.append({
            "uid": uid, "x": lr, "v": v24, "mean": mean_lr,
            "sigma": sigma, "play_v": play_v, "expiry": expiry_ts,
        })

    rng = random.Random(SEED)
    steps = int(horizon_h // STEP_H)
    if steps < 1:
        steps = 1
    counts = {}
    win_from_ts = win_from.timestamp() if win_from else last_ts
    for _trial in range(TRIALS):
        xs = [s["x"] for s in states]
        for step in range(steps):
            elapsed = step * STEP_H
            t_now = win_from_ts + elapsed * 3600.0
            decay = math.exp(-elapsed / 24.0)
            for i, s in enumerate(states):
                if s["expiry"] is not None and t_now >= s["expiry"]:
                    xs[i] = 999.0
                    continue
                if xs[i] >= 900.0:
                    continue
                v_eff = s["v"] * decay + s["play_v"] * decay
                rev = (0.114 - 0.033 * xs[i]) * (STEP_H / 6.0) * 0.5
                # Student-t innovation (df 5), rescaled to unit variance
                # (var t_df = df/(df-2)); heavy tails match the observed
                # log-rank kurtosis (~15.8).
                z = rng.gauss(0.0, 1.0)
                chi = rng.gammavariate(T_DF / 2.0, 2.0)
                noise = (z / math.sqrt(chi / T_DF)
                         / math.sqrt(T_DF / (T_DF - 2.0))
                         * s["sigma"] * math.sqrt(STEP_H))
                xs[i] = xs[i] + v_eff * STEP_H + rev + noise
            order = sorted(range(len(uids)), key=lambda i: xs[i])
            for place_idx, gi in enumerate(order[:places]):
                if xs[gi] >= 900.0:
                    continue
                key = (place_idx + 1, uids[gi])
                counts[key] = counts.get(key, 0) + 1
    total = TRIALS * steps
    probs = []
    for place in range(1, places + 1):
        entries = []
        for uid in uids:
            c = counts.get((place, uid), 0)
            if c > 0:
                entries.append((uid, c / total))
        entries.sort(key=lambda x: (-x[1], x[0]))
        capped = [(u, min(p, CAP)) for u, p in entries]
        ssum = sum(p for _, p in capped)
        if ssum > SUM_TARGET:
            scale = SUM_TARGET / ssum
            capped = [(u, p * scale) for u, p in capped]
        for uid, p in capped:
            if p >= 0.002:
                probs.append({"universe_id": uid, "place": place, "p": round(p, 4)})

    # Breakout blend: reallocate a share of each place's mass to games
    # whose playing is surging / that climbed hard over the last 72h.
    if breakout:
        pv = {1: 0.35, 2: 0.35, 3: 0.30, 4: 0.25, 5: 0.25,
              6: 0.20, 7: 0.20, 8: 0.15, 9: 0.15, 10: 0.15}
        by_place = {}
        for row in probs:
            by_place.setdefault(row["place"], {})[row["universe_id"]] = row["p"]
        total_score = sum(breakout.values())
        new_probs = []
        for place in range(1, places + 1):
            share = pv.get(place, 0.2) * BREAKOUT_SCALE
            named = by_place.get(place, {})
            ssum = sum(named.values())
            new = {uid: p * (1.0 - share) for uid, p in named.items()}
            for uid, sc in breakout.items():
                new[uid] = new.get(uid, 0.0) + share * ssum * (sc / total_score)
            for uid, p in sorted(new.items(), key=lambda kv: (-kv[1], kv[0])):
                if p >= 0.002:
                    new_probs.append({"universe_id": uid, "place": place,
                                      "p": round(p, 4)})
        probs = new_probs

    # Top 10 by blended probability, weighted toward the top places.
    score10 = {}
    for row in probs:
        score10[row["universe_id"]] = score10.get(row["universe_id"], 0.0) \
            + row["p"] * (places + 1 - row["place"])
    top10 = sorted(score10, key=lambda u: (-score10[u], u))[:places]
    # Fall back to the last chart's order if anything went wrong.
    if len(top10) < places:
        for uid in uids:
            if uid not in top10:
                top10.append(uid)
            if len(top10) >= places:
                break

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
        f.write(json.dumps({"probabilities": probs, "top10": top10}) + "\n")


if __name__ == "__main__":
    main()
