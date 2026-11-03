"""Playing styles and matchup history.

Each player's style comes from decayed serve/return stats: big server (aces and serve points well above the tour),
returner (return points well above), or balanced; crossed with lefty/righty. For every match we record how far each
player beat their rating's expectation, split by the opponent's style, and the same for head-to-head meetings.
Feature for A vs B = how A has done against B's style (beyond expectation) minus how B has done against A's style.
"""
HL = 540.0   # days


class Dec:
    __slots__ = ("x", "w", "t")

    def __init__(self):
        self.x = self.w = 0.0
        self.t = 0

    def add(self, x, w, t):
        if self.t and t > self.t:
            f = 0.5 ** ((t - self.t) / HL)
            self.x *= f; self.w *= f
        self.t = max(self.t, t)
        self.x += x; self.w += w

    def val(self, t):
        if not self.t:
            return 0.0, 0.0
        f = 0.5 ** (max(t - self.t, 0) / HL)
        return self.x * f, self.w * f


class Style:
    def __init__(self):
        self.ace = {}         # key -> Dec of aces per serve point
        self.spw = {}         # key -> Dec of serve points won
        self.rpw = {}         # key -> Dec of return points won
        self.lefty = {}       # key -> bool
        self.tour = {}        # tour -> [Dec ace, Dec spw, Dec rpw] means
        self.vs = {}          # (key, style) -> Dec of (result - expected)
        self.h2h = {}         # (a, b) -> Dec of (result - expected) for a

    def _d(self, m, k):
        if k not in m:
            m[k] = Dec()
        return m[k]

    def style(self, k, tour, t):
        """'S' big server, 'R' returner, 'B' balanced; + 'L'/'R' hand. None while there is too little data."""
        a, aw = self._d(self.ace, k).val(t)
        s, sw = self._d(self.spw, k).val(t)
        r, rw = self._d(self.rpw, k).val(t)
        if sw < 300:
            return None
        T = self.tour.get(tour)
        if not T:
            return None
        ta, taw = T[0].val(t); ts, tsw = T[1].val(t); tr, trw = T[2].val(t)
        if taw <= 0 or tsw <= 0 or trw <= 0:
            return None
        ace, sp, rp = a / aw, s / sw, r / rw
        mace, msp, mrp = ta / taw, ts / tsw, tr / trw
        if ace > mace * 1.35 and sp > msp + 0.01:
            st = "S"
        elif rp > mrp + 0.02 and ace < mace * 1.1:
            st = "R"
        else:
            st = "B"
        return st + ("L" if self.lefty.get(k) else "R")

    def features(self, a, b, tour, t):
        sa, sb = self.style(a, tour, t), self.style(b, tour, t)
        out = {"style_a": sa, "style_b": sb, "vs_a": 0.0, "vs_b": 0.0, "vs_n_a": 0.0, "vs_n_b": 0.0, "h2h_res": 0.0, "h2h_n": 0.0}
        if sb:
            x, w = self._d(self.vs, (a, sb)).val(t)
            out["vs_a"], out["vs_n_a"] = x / (w + 8), w
        if sa:
            x, w = self._d(self.vs, (b, sa)).val(t)
            out["vs_b"], out["vs_n_b"] = x / (w + 8), w
        x, w = self._d(self.h2h, (a, b)).val(t)
        out["h2h_res"], out["h2h_n"] = x / (w + 3), w
        return out

    def update(self, m, t, p_w):
        """m: the match row (winner m['w'], loser m['l'], optional stats); p_w: expected win prob of the winner."""
        w, l, tour = m["w"], m["l"], m["tour"]
        for side, k in (("w", w), ("l", l)):
            h = m.get(f"{side}_hand")
            if h:
                self.lefty[k] = h == "L"
        sw_, sl_ = self.style(w, tour, t), self.style(l, tour, t)
        if p_w is not None:
            if sl_:
                self._d(self.vs, (w, sl_)).add(1 - p_w, 1, t)
            if sw_:
                self._d(self.vs, (l, sw_)).add(-(1 - p_w), 1, t)
            self._d(self.h2h, (w, l)).add(1 - p_w, 1, t)
            self._d(self.h2h, (l, w)).add(-(1 - p_w), 1, t)
        try:
            wsv, w1, w2, wa = float(m["w_svpt"]), float(m["w_1stWon"]), float(m["w_2ndWon"]), float(m.get("w_ace") or 0)
            lsv, l1, l2, la = float(m["l_svpt"]), float(m["l_1stWon"]), float(m["l_2ndWon"]), float(m.get("l_ace") or 0)
        except (KeyError, TypeError, ValueError):
            return
        if wsv < 20 or lsv < 20:
            return
        T = self.tour.setdefault(tour, [Dec(), Dec(), Dec()])
        for k, sv, won, ace, opp_sv, opp_won in ((w, wsv, w1 + w2, wa, lsv, l1 + l2), (l, lsv, l1 + l2, la, wsv, w1 + w2)):
            self._d(self.ace, k).add(ace, sv, t)
            self._d(self.spw, k).add(won, sv, t)
            self._d(self.rpw, k).add(opp_sv - opp_won, opp_sv, t)
            T[0].add(ace, sv, t); T[1].add(won, sv, t); T[2].add(opp_sv - opp_won, opp_sv, t)
