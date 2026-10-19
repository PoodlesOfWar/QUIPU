"""Check qpsi.hidden_work's Φ(δt) estimator against the data of Muruga et al., Nature Physics (2026),
doi:10.1038/s41567-026-03442-w (data and MATLAB code: doi:10.5281/zenodo.19881972, CC BY 4.0).

Uses the paper's Raw_Data folder (9 one-hour equilibrium trajectories at 100 fps, in Lokesh_Information_Data.zip)
and the conventions of its Data_Analyzer.m: x in nm (pixel × 16, mean removed per file), threshold x_th = 75 nm,
a 3 s relaxation window, energy E = x²/(2⟨x²⟩) in k_BT. The estimator is the one in
qpsi.hidden_work.memory_order_parameter, written with numpy so the full 3.24 million frames can be used:
outcome-resolved KL relaxation against the stationary distribution (12 equal-probability bins), minus a history-
blind control drawn from the single-measurement ensemble with matched initial bins (eq. 4).

    python bench/validate_hidden_work.py <Raw_Data folder> [out.json] [δt seconds ...]

Needs numpy and h5py (the .mat files are MATLAB v7.3). Not part of the test suite.
"""
import glob
import json
import sys
import time

import h5py
import numpy as np

PX_NM, XTH_NM, FPS, TREL = 16.0, 75.0, 100, 300


def load(folder):
    xs = []
    for f in sorted(glob.glob(folder.rstrip("/\\") + "/*.mat")):
        with h5py.File(f) as h:
            pos = np.array(h["pos"]).T                 # stored transposed: x, y, frame
        xs.append((pos[:, 0] - pos[:, 0].mean()) * PX_NM)
    return xs


def run(xs, deltas, nmax=4000, draws=3, seed=7):
    allx = np.concatenate(xs)
    kap = 1.0 / np.mean(allx ** 2)
    edges = np.quantile(allx, np.linspace(0, 1, 13)[1:-1])
    nb = len(edges) + 1
    eq = np.bincount(np.searchsorted(edges, allx, side="right"), minlength=nb).astype(float)
    q = (eq + 0.5) / (eq.sum() + 0.5 * nb)
    rng = np.random.default_rng(seed)

    def at(fi, st, t):
        return np.concatenate([xs[k][st[fi == k] + t] for k in np.unique(fi)])

    def kl(fi, st):
        out = np.empty(TREL + 1)
        for t in range(TREL + 1):
            c = np.bincount(np.searchsorted(edges, at(fi, st, t), side="right"), minlength=nb)
            p = (c + 0.5) / (c.sum() + 0.5 * nb)
            out[t] = np.sum(p * np.log(p / q))
        return out

    rows = []
    for delta_s in deltas:
        d = int(round(delta_s * FPS))
        F, S, I, J, B = [], [], [], [], []
        for k, x in enumerate(xs):
            s = np.arange(max(d, 0), len(x) - TREL)
            F.append(np.full(s.size, k)); S.append(s)
            I.append(x[s - d] > XTH_NM); J.append(x[s] > XTH_NM)
            B.append(np.searchsorted(edges, x[s], side="right"))
        F, S, I, J, B = map(np.concatenate, (F, S, I, J, B))
        phi, states = 0.0, {}
        for i in (0, 1):
            for j in (0, 1):
                m = np.flatnonzero((I == i) & (J == j))
                if m.size < 50:
                    continue
                p = m.size / S.size
                pick = rng.choice(m, size=min(nmax, m.size), replace=False)
                real = kl(F[pick], S[pick])
                energy = np.array([0.5 * kap * np.mean(at(F[pick], S[pick], t) ** 2) for t in range(TREL + 1)])
                pool = np.flatnonzero(J == j)
                ctl = []
                for _ in range(draws):
                    chosen = np.concatenate([rng.choice(pool[B[pool] == b], size=int(np.sum(B[pick] == b)))
                                             for b in np.unique(B[pick])])
                    ctl.append(kl(F[chosen], S[chosen]))
                ph = float(np.sum(real - np.mean(ctl, axis=0)) / FPS)
                states[f"|{i}{j}⟩"] = {"p": round(p, 4), "n": int(m.size), "phi": round(ph, 4),
                                       "E0_kT": round(float(energy[0]), 3), "Emax_kT": round(float(energy.max()), 3),
                                       "E3s_kT": round(float(energy[-1]), 3)}
                phi += p * ph
        rows.append({"delta_s": delta_s, "phi": round(phi, 5), "states": states})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return {"frames": int(allx.size), "hours": round(allx.size / FPS / 3600, 2), "threshold_nm": XTH_NM,
            "window_s": TREL / FPS, "rows": rows, "computed": time.strftime("%Y-%m-%d")}


if __name__ == "__main__":
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".json") else None
    deltas = [float(v) for v in sys.argv[3 if out else 2:]] or [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 200]
    doc = run(load(folder), deltas)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
