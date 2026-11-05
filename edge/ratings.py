"""Walk-forward player ratings: overall + surface Elo (margin-weighted), opponent-adjusted serve/return point
rates with time decay, form residual, fatigue and layoff. Pure Python so the cloud refresher needs no numpy.

Feed matches in date order through `Ratings.add(m)`; `Ratings.features(a, b, ctx)` gives the pre-match numbers
the model uses. State round-trips through `dump()` / `load()` so the cloud only replays recent results.
"""
import math
import re
from datetime import date

SURFACES = ("Hard", "Clay", "Grass", "Carpet")
HALF_LIFE = 300.0          # days, serve/return stats
FORM_HL = 45.0             # days, form residual
SHRINK_PTS = 900.0         # pseudo serve points pulling a player's rates to the tour mean
# Tuned 2026-11-04 (tools/elo_tune, walk-forward on 1.6M matches; chosen on ITF 2016-21, checked on 2022-26):
# faster learning (K0 400, offset 2), scorelines count 1.5x more, Challenger/qualifying results count 80%.
# ITF test log-loss 0.5641 -> 0.5600, all levels 0.5760 -> 0.5716. Bump RATINGS_VER whenever these change so the
# cloud swaps its rolling state for the rebuilt seed (edge/refresh.load_state).
RATINGS_VER = 2
LEVEL_K = {"G": 1.0, "M": 1.0, "F": 1.0, "A": 1.0, "C": 0.8, "S": 0.85, "D": 0.85, "Q": 0.8, "O": 0.85}
MARGIN = 1.5


def ordinal(d):
    d = str(d)
    return date(int(d[:4]), int(d[4:6]), int(d[6:8])).toordinal()


def parse_score(s):
    """'6-4 3-6 7-6(4)' -> (winner games, loser games, sets played, tiebreaks, retired)."""
    s = s or ""
    ret = bool(re.search(r"RET|DEF|ABD|ABN|Def", s))
    gw = gl = sets = tbs = 0
    for a, b in re.findall(r"(\d+)-(\d+)", re.sub(r"\(\d+\)", "", s)):
        a, b = int(a), int(b)
        if a > 13 or b > 13:          # match tiebreak written as 10-8
            a, b = (1, 0) if a > b else (0, 1)
        gw += a; gl += b; sets += 1
        tbs += (a == 7 and b == 6) or (a == 6 and b == 7)
    return gw, gl, sets, tbs, ret


class Dec:
    """Exponentially decayed running sum of (value, weight)."""
    __slots__ = ("x", "w", "t")

    def __init__(self, x=0.0, w=0.0, t=0):
        self.x, self.w, self.t = x, w, t

    def add(self, x, w, t, hl):
        if self.t and t > self.t:
            f = 0.5 ** ((t - self.t) / hl)
            self.x *= f; self.w *= f
        self.t = max(self.t, t)
        self.x += x; self.w += w

    def at(self, t, hl):
        f = 0.5 ** (max(t - self.t, 0) / hl) if self.t else 0
        return self.x * f, self.w * f


class Player:
    __slots__ = ("name", "elo", "n", "selo", "sn", "last", "sv", "rt", "ssv", "srt", "form", "recent", "hand", "ht", "ioc", "born", "rank", "wins", "losses")

    def __init__(self, name):
        self.name = name
        self.elo, self.n = 1500.0, 0
        self.selo = {s: 1500.0 for s in SURFACES}
        self.sn = {s: 0 for s in SURFACES}
        self.last = 0
        self.sv, self.rt = Dec(), Dec()                     # adjusted serve / return points won (decayed)
        self.ssv = {s: Dec() for s in SURFACES}
        self.srt = {s: Dec() for s in SURFACES}
        self.form = Dec()                                   # sum of (result - expected)
        self.recent = []                                    # [(ordinal, minutes, won, opp_name, surface, score)] last 12
        self.hand = self.ht = self.ioc = self.born = None
        self.rank = None
        self.wins = self.losses = 0


def k_factor(n):
    return 400.0 / (n + 2) ** 0.4


def p_elo(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


class Ratings:
    def __init__(self, w_surface=0.5, margin=True):
        self.P = {}
        self.w_surface = w_surface
        self.margin = margin
        # tour x surface mean serve-point win rate, decayed so it follows the era
        self.mean = {}
        self.h2h = {}
        self.seen = set()

    # ---------- helpers
    def player(self, key, name=None):
        p = self.P.get(key)
        if p is None:
            p = self.P[key] = Player(name or key)
        return p

    def tour_mean(self, tour, surface, t):
        d = self.mean.get((tour, surface))
        if not d:
            return 0.64 if tour == "atp" else 0.565
        x, w = d.at(t, 720)
        return x / w if w > 50 else (0.64 if tour == "atp" else 0.565)

    def blended(self, p, surface):
        if surface in p.selo:
            w = self.w_surface * min(1.0, p.sn[surface] / 10.0)
            return (1 - w) * p.elo + w * p.selo[surface]
        return p.elo

    def serve_rates(self, p, tour, surface, t):
        """Shrunk serve and return points-won rates (overall blended with surface), and the evidence weight."""
        m = self.tour_mean(tour, surface, t)
        sx, sw = p.sv.at(t, HALF_LIFE)
        rx, rw = p.rt.at(t, HALF_LIFE)
        if surface in p.ssv:
            ssx, ssw = p.ssv[surface].at(t, HALF_LIFE)
            srx, srw = p.srt[surface].at(t, HALF_LIFE)
            sx, sw = sx + ssx, sw + ssw            # surface points count double
            rx, rw = rx + srx, rw + srw
        spw = (sx + SHRINK_PTS * m) / (sw + SHRINK_PTS)
        rpw = (rx + SHRINK_PTS * (1 - m)) / (rw + SHRINK_PTS)
        return spw, rpw, sw

    # ---------- features for a matchup (A vs B), all from A's side
    def features(self, ka, kb, tour, surface, t, best_of=3, level="A"):
        A, B = self.P.get(ka), self.P.get(kb)
        if A is None or B is None:
            return None
        ea, eb = self.blended(A, surface), self.blended(B, surface)
        m = self.tour_mean(tour, surface, t)
        sa, ra, wa = self.serve_rates(A, tour, surface, t)
        sb, rb, wb = self.serve_rates(B, tour, surface, t)
        pa = min(max(m + (sa - m) - (rb - (1 - m)), 0.3), 0.92)       # A serving vs B returning
        pb = min(max(m + (sb - m) - (ra - (1 - m)), 0.3), 0.92)
        fa = A.form.at(t, FORM_HL)
        fb = B.form.at(t, FORM_HL)
        h = self.h2h.get((ka, kb), 0) - self.h2h.get((kb, ka), 0)
        hn = self.h2h.get((ka, kb), 0) + self.h2h.get((kb, ka), 0)
        return {
            "p_elo": p_elo(ea, eb), "p_elo_all": p_elo(A.elo, B.elo),
            "elo_a": ea, "elo_b": eb, "n_a": A.n, "n_b": B.n,
            "pa": pa, "pb": pb, "stat_w": min(wa, wb), "mean_spw": m,
            "form_a": fa[0] / (fa[1] + 2), "form_b": fb[0] / (fb[1] + 2),
            "days_a": t - A.last if A.last else 999, "days_b": t - B.last if B.last else 999,
            "load_a": self.minutes_load(A, t), "load_b": self.minutes_load(B, t),
            "h2h": h, "h2h_n": hn,
        }

    @staticmethod
    def minutes_load(p, t, days=10):
        """Minutes played in the last `days` days (90 per match when unknown)."""
        return sum((mn or 90) for (d, mn, *_r) in p.recent if 0 <= t - d <= days)

    # ---------- update
    def add(self, m):
        """m: dict with tour, date (yyyymmdd or ordinal), surface, level, w, l (keys), w_name, l_name, score,
        minutes, best_of, and optional stats w_svpt w_1stWon w_2ndWon l_svpt l_1stWon l_2ndWon."""
        t = m["date"] if isinstance(m["date"], int) and m["date"] > 700000 else ordinal(m["date"])
        uid = m.get("uid")
        if uid:
            if uid in self.seen:
                return
            self.seen.add(uid)
        W, L = self.player(m["w"], m.get("w_name")), self.player(m["l"], m.get("l_name"))
        surface = m.get("surface") if m.get("surface") in SURFACES else None
        gw, gl, sets, tbs, ret = parse_score(m.get("score"))
        if "W/O" in (m.get("score") or "") or (ret and gw + gl < 6):
            return
        lvl = LEVEL_K.get(m.get("level"), 0.9)
        weight = lvl * (0.5 if ret else 1.0)
        # margin multiplier (Weighted Elo): dominant wins move ratings more
        mult = 1.0
        if self.margin and gw + gl > 0 and not ret:
            share = gw / (gw + gl)                          # winner's share of games, ~0.5..1
            mult = max(0.3, 1 + MARGIN * 1.6 * (share - 0.75))   # 0.4 for a coin-flip win, 1.6 for a bagel-bagel
        e = p_elo(W.elo, L.elo)
        kw, kl = k_factor(W.n), k_factor(L.n)
        lay_w = 1.3 if W.last and t - W.last > 120 else 1.0
        lay_l = 1.3 if L.last and t - L.last > 120 else 1.0
        W.elo += kw * weight * mult * lay_w * (1 - e)
        L.elo -= kl * weight * mult * lay_l * (1 - e)
        if surface:
            es = p_elo(W.selo[surface], L.selo[surface])
            W.selo[surface] += k_factor(W.sn[surface]) * weight * mult * (1 - es)
            L.selo[surface] -= k_factor(L.sn[surface]) * weight * mult * (1 - es)
            W.sn[surface] += 1; L.sn[surface] += 1
        # form residual vs blended expectation
        eb = p_elo(self.blended(W, surface), self.blended(L, surface))
        W.form.add(1 - eb, 1, t, FORM_HL)
        L.form.add(-(1 - eb), 1, t, FORM_HL)
        W.n += 1; L.n += 1
        W.wins += 1; L.losses += 1
        # serve / return stats, opponent-adjusted
        try:
            wsv, w1, w2 = float(m["w_svpt"]), float(m["w_1stWon"]), float(m["w_2ndWon"])
            lsv, l1, l2 = float(m["l_svpt"]), float(m["l_1stWon"]), float(m["l_2ndWon"])
            ok = wsv > 20 and lsv > 20 and not ret
        except (KeyError, TypeError, ValueError):
            ok = False
        if ok:
            tour = m["tour"]
            mu = self.tour_mean(tour, surface or "Hard", t)
            w_spw, l_spw = (w1 + w2) / wsv, (l1 + l2) / lsv
            # opponent adjustment: a serve rate against a strong returner counts for more
            _, w_rpw0, _ = self.serve_rates(W, tour, surface or "Hard", t)
            l_spw0, l_rpw0, _ = self.serve_rates(L, tour, surface or "Hard", t)
            w_spw0, _, _ = self.serve_rates(W, tour, surface or "Hard", t)
            adj_w_sv = w_spw + (l_rpw0 - (1 - mu))
            adj_l_sv = l_spw + (w_rpw0 - (1 - mu))
            adj_w_rt = (1 - l_spw) + (l_spw0 - mu)
            adj_l_rt = (1 - w_spw) + (w_spw0 - mu)
            W.sv.add(adj_w_sv * wsv, wsv, t, HALF_LIFE); L.sv.add(adj_l_sv * lsv, lsv, t, HALF_LIFE)
            W.rt.add(adj_w_rt * lsv, lsv, t, HALF_LIFE); L.rt.add(adj_l_rt * wsv, wsv, t, HALF_LIFE)
            if surface:
                W.ssv[surface].add(adj_w_sv * wsv, wsv, t, HALF_LIFE); L.ssv[surface].add(adj_l_sv * lsv, lsv, t, HALF_LIFE)
                W.srt[surface].add(adj_w_rt * lsv, lsv, t, HALF_LIFE); L.srt[surface].add(adj_l_rt * wsv, wsv, t, HALF_LIFE)
            d = self.mean.setdefault((tour, surface or "Hard"), Dec())
            d.add((w1 + w2 + l1 + l2), wsv + lsv, t, 720)
        mins = None
        try:
            mins = float(m.get("minutes")) if m.get("minutes") not in (None, "") else None
        except ValueError:
            pass
        for P_, won, opp in ((W, 1, L.name), (L, 0, W.name)):
            P_.recent.append((t, mins, won, opp, surface, m.get("score") or ""))
            if len(P_.recent) > 12:
                del P_.recent[0]
            P_.last = max(P_.last, t)
        self.h2h[(m["w"], m["l"])] = self.h2h.get((m["w"], m["l"]), 0) + 1
        for side, P_ in (("w", W), ("l", L)):
            for f, attr in (("hand", "hand"), ("ht", "ht"), ("ioc", "ioc"), ("rank", "rank")):
                v = m.get(f"{side}_{f}")
                if v not in (None, ""):
                    setattr(P_, attr, v)
            v = m.get(f"{side}_age")
            if v not in (None, ""):
                try:
                    P_.born = t - float(v) * 365.25
                except ValueError:
                    pass

    # ---------- persistence
    def dump(self, min_last=0):
        def dec(d):
            return [round(d.x, 3), round(d.w, 3), d.t]
        out = {"w_surface": self.w_surface, "margin": self.margin, "mean": {f"{a}|{b}": dec(v) for (a, b), v in self.mean.items()},
               "players": {}, "h2h": {}}
        keep = set()
        for k, p in self.P.items():
            if p.last < min_last:
                continue
            keep.add(k)
            out["players"][k] = [p.name, round(p.elo, 2), p.n, {s: [round(p.selo[s], 2), p.sn[s]] for s in SURFACES if p.sn[s]},
                                 p.last, dec(p.sv), dec(p.rt), {s: [dec(p.ssv[s]), dec(p.srt[s])] for s in SURFACES if p.ssv[s].w},
                                 dec(p.form), [list(r) for r in p.recent], p.hand, p.ht, p.ioc, p.born, p.rank, p.wins, p.losses]
        out["h2h"] = {f"{a}~{b}": n for (a, b), n in self.h2h.items() if a in keep and b in keep}
        return out

    @classmethod
    def load(cls, d):
        r = cls(d["w_surface"], d["margin"])
        r.mean = {tuple(k.split("|")): Dec(*v) for k, v in d["mean"].items()}
        for k, v in d["players"].items():
            p = Player(v[0])
            p.elo, p.n = v[1], v[2]
            for s, (e, n) in v[3].items():
                p.selo[s], p.sn[s] = e, n
            p.last = v[4]
            p.sv, p.rt = Dec(*v[5]), Dec(*v[6])
            for s, (a, b) in v[7].items():
                p.ssv[s], p.srt[s] = Dec(*a), Dec(*b)
            p.form = Dec(*v[8])
            p.recent = [tuple(x) for x in v[9]]
            p.hand, p.ht, p.ioc, p.born, p.rank, p.wins, p.losses = v[10:17]
            r.P[k] = p
        r.h2h = {tuple(k.split("~")): n for k, n in d["h2h"].items()}
        return r
