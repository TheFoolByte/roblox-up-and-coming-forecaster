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
SEED = int(os.environ.get("MF_SEED", "20261112"))
TRIALS = int(os.environ.get("MF_TRIALS", "1500"))
STEP_H = float(os.environ.get("MF_STEP_H", "3.0"))
CAP = float(os.environ.get("MF_CAP", "0.25"))
SUM_TARGET = float(os.environ.get("MF_SUM_TARGET", "0.85"))
TEMP = float(os.environ.get("MF_TEMP", "1.0"))
VEL_CLAMP = float(os.environ.get("MF_VEL_CLAMP", "0.02"))
VEL_WINDOW = float(os.environ.get("MF_VEL_WINDOW", "24"))
PLAY_DRIFT = float(os.environ.get("MF_PLAY_DRIFT", "1.0"))
PLAY_CLAMP = float(os.environ.get("MF_PLAY_CLAMP", "0.04"))
DEBUT_BONUS = float(os.environ.get("MF_DEBUT", "0.5"))
DEBUT_H = float(os.environ.get("MF_DEBUT_H", "48"))
TENURE_DAYS = float(os.environ.get("MF_TENURE_DAYS", "28"))
DRIFT_HL = float(os.environ.get("MF_DRIFT_HALF_LIFE", "24.0"))
BLEND = float(os.environ.get("MF_BLEND", "0.05"))  # v4: 5% blend toward persistence forecast (round 3, verified +0.622 late)
SIGMA_MID = float(os.environ.get("MF_SIGMA_MID", "1.0"))
SIGMA_SCALE = float(os.environ.get("MF_SIGMA_SCALE", "1.0"))
CLIMB_PBOOST = float(os.environ.get("MF_CLIMB_PBOOST", "1.0"))
CLIMB_VMULT = float(os.environ.get("MF_CLIMB_VMULT", "1.0"))
CLIMB_VTHRESH = float(os.environ.get("MF_CLIMB_VTHRESH", "-0.01"))
CLIMB_RMIN = int(os.environ.get("MF_CLIMB_RMIN", "11"))
RENORM = int(os.environ.get("MF_RENORM", "1"))  # v4: redistribute capped excess instead of discarding (round 3)
CAP1 = float(os.environ.get("MF_CAP1", "0"))
VEL_SHRINK = float(os.environ.get("MF_VEL_SHRINK", "0.3"))  # v4: damp noisy 24h velocity to 30% (round 3, verified +0.622 late)
# Tuned (v3 aggressive): Student-t innovations (df 5) match the heavy tails of
# log-rank moves (kurtosis ~15.8); breakout blend at 0.6x — smaller than the
# safe 0.75x, a bet that the final window behaves like the late rolling cuts
# (where 0.6x won biggest), at the cost of ~0.13 practice score.
T_DF = float(os.environ.get("MF_T_DF", "5.0"))
BREAKOUT_SCALE = float(os.environ.get("MF_BREAKOUT_SCALE", "0.6"))
REV_SCALE = float(os.environ.get("MF_MEAN_REV", "0.5"))
HORIZON_K = float(os.environ.get("MF_SIGMA_HORIZON", "0"))


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
                   if r and last_ts - t.timestamp() <= VEL_WINDOW * 3600.0]
        if len(win_pts) >= 2:
            xs = [x for x, _ in win_pts]
            ys = [y for _, y in win_pts]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            if den > 0:
                # slope per second -> per hour
                v24 = sum((x - mx) * (y - my) for x, y in win_pts) / den * 3600.0
        v24_raw = v24
        v24 = max(-VEL_CLAMP, min(VEL_CLAMP, v24 * VEL_SHRINK))
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
        play_v = -0.01 * PLAY_DRIFT * (g24 + g48)
        play_v = max(-PLAY_CLAMP, min(PLAY_CLAMP, play_v))
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
        if 11 <= r0 <= 50:
            sigma *= SIGMA_MID
        sigma *= SIGMA_SCALE
        first_t = pts[0][0].timestamp() if pts else last_ts
        if last_ts - first_t < 48 * 3600.0:
            sigma *= 1.2
        # Tenure hazard: created + 28 days.
        expiry_ts = None
        det = details.get(uid) or {}
        created = parse_time(det.get("created")) if det else None
        if created is not None and TENURE_DAYS > 0:
            expiry_ts = created.timestamp() + TENURE_DAYS * 86400.0
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
        if last_ts - first_t < DEBUT_H * 3600.0 and (g_step > 0.5 or g24 > 0.8):
            score += DEBUT_BONUS
        if score > 0:
            breakout[uid] = score
        states.append({
            "uid": uid, "x": lr, "v": v24, "mean": mean_lr,
            "sigma": sigma, "play_v": play_v, "expiry": expiry_ts,
            "r0": r0, "v_raw": v24_raw,
            "climb": (r0 >= CLIMB_RMIN and v24_raw < CLIMB_VTHRESH),
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
            decay = math.exp(-elapsed / DRIFT_HL)
            for i, s in enumerate(states):
                if s["expiry"] is not None and t_now >= s["expiry"]:
                    xs[i] = 999.0
                    continue
                if xs[i] >= 900.0:
                    continue
                v_base = s["v"] * (CLIMB_VMULT if s.get("climb") else 1.0)
                v_eff = v_base * decay + s["play_v"] * decay
                rev = (0.114 - 0.033 * xs[i]) * (STEP_H / 6.0) * REV_SCALE
                # Student-t innovation (df 5), rescaled to unit variance
                # (var t_df = df/(df-2)); heavy tails match the observed
                # log-rank kurtosis (~15.8).
                z = rng.gauss(0.0, 1.0)
                chi = rng.gammavariate(T_DF / 2.0, 2.0)
                noise = (z / math.sqrt(chi / T_DF)
                         / math.sqrt(T_DF / (T_DF - 2.0))
                         * s["sigma"] * (1.0 + HORIZON_K * elapsed / 72.0) * math.sqrt(STEP_H))
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
        cap_here = CAP1 if (place == 1 and CAP1 > 0) else CAP
        capped = [(u, min(p, cap_here)) for u, p in entries]
        ssum = sum(p for _, p in capped)
        if ssum > SUM_TARGET:
            scale = SUM_TARGET / ssum
            capped = [(u, p * scale) for u, p in capped]
        elif RENORM and ssum > 0:
            scale = SUM_TARGET / ssum
            capped = [(u, p * scale) for u, p in capped]
        if TEMP != 1.0:
            tsum = sum(p for _, p in capped)
            if tsum > 0:
                powered = [(u, p ** TEMP) for u, p in capped]
                psum = sum(p for _, p in powered)
                capped = [(u, p * tsum / psum) for u, p in powered]
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

    # Targeted climber boost: multiply the final per-place probability of
    # rank-11+ games with strong upward velocity, renormalizing each place
    # to its pre-boost sum (mass comes from non-climbers at the same place).
    if CLIMB_PBOOST != 1.0:
        climber_uids = {s["uid"] for s in states if s.get("climb")}
        if climber_uids:
            by_place3 = {}
            for row in probs:
                by_place3.setdefault(row["place"], {})[row["universe_id"]] = row["p"]
            boosted = []
            for place in range(1, places + 1):
                named = by_place3.get(place, {})
                ssum = sum(named.values())
                new = {uid: (p * CLIMB_PBOOST if uid in climber_uids else p)
                       for uid, p in named.items()}
                nsum = sum(new.values())
                if nsum > 0:
                    new = {uid: p * ssum / nsum for uid, p in new.items()}
                for uid, p in sorted(new.items(), key=lambda kv: (-kv[1], kv[0])):
                    if p >= 0.002:
                        boosted.append({"universe_id": uid, "place": place,
                                        "p": round(p, 4)})
            probs = boosted

    # Persistence blend (house-baseline style): per place, 0.30 on the
    # incumbent, 0.10 on each neighbour; mixed in with weight BLEND,
    # preserving each place's named sum.
    if BLEND > 0:
        by_place2 = {}
        for row in probs:
            by_place2.setdefault(row["place"], {})[row["universe_id"]] = row["p"]
        blended = []
        for place in range(1, places + 1):
            named = by_place2.get(place, {})
            ssum = sum(named.values())
            pers = {}
            if place - 1 < len(uids):
                pers[uids[place - 1]] = 0.30
            if place >= 2:
                pers[uids[place - 2]] = pers.get(uids[place - 2], 0.0) + 0.10
            if place < len(uids):
                pers[uids[place]] = pers.get(uids[place], 0.0) + 0.10
            psum = sum(pers.values())
            new = {}
            for uid, pv in pers.items():
                new[uid] = BLEND * ssum * pv / psum if psum > 0 else 0.0
            for uid, pv in named.items():
                new[uid] = new.get(uid, 0.0) + pv * (1.0 - BLEND)
            for uid, pv in sorted(new.items(), key=lambda kv: (-kv[1], kv[0])):
                if pv >= 0.002:
                    blended.append({"universe_id": uid, "place": place,
                                    "p": round(pv, 4)})
        probs = blended

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
