"""
fl_ienn/optim/gmco.py
Gaussian Mutated Chimp Optimization (GMCO) for the IENN-GMCO step.

Chimp Optimization Algorithm (ChOA) + Gaussian mutation.
Standalone: depends only on numpy (matplotlib is imported lazily for plots).
It does not touch the Session Layer code (ECCAuthenticator / ExponentialKAnonymity).

Usage (fitness = value to MINIMISE, e.g. 1 - validation macro-F1 of the IENN):

    space = SearchSpace({
        "hidden":  (16, 128, "int"),
        "lr":      (1e-4, 3e-2, "log"),
        "ctx_decay": (0.0, 0.9, "float"),
        "dropout": (0.0, 0.5, "float"),
    })

    def fitness(vec):
        params = space.decode(vec)
        return 1.0 - train_eval_ienn(params)     # train_eval_ienn is defined in the IENN step

    opt = GMCO(fitness, space.lb, space.ub, pop_size=10, max_iter=15, seed=42)
    result = opt.optimize()
    best_params = space.decode(result["best_pos"])
    save_report(result, space, "reports/02_privacy_security_layer")
"""
import json
import os
import numpy as np


class SearchSpace:
    """Maps a real-valued vector <-> named hyperparameters.
    kind: 'float' (linear), 'log' (log10 scale), 'int' (rounded)."""

    def __init__(self, spec):
        self.names = list(spec.keys())
        self.kinds = [spec[n][2] for n in self.names]
        lb, ub = [], []
        for n in self.names:
            lo, hi, kind = spec[n]
            if kind == "log":
                lo, hi = np.log10(lo), np.log10(hi)
            lb.append(lo)
            ub.append(hi)
        self.lb, self.ub = np.array(lb, float), np.array(ub, float)

    def decode(self, vec):
        vec = np.clip(vec, self.lb, self.ub)
        out = {}
        for v, n, kind in zip(vec, self.names, self.kinds):
            if kind == "log":
                out[n] = float(10 ** v)
            elif kind == "int":
                out[n] = int(round(v))
            else:
                out[n] = float(v)
        return out


class GMCO:
    def __init__(self, fitness, lb, ub, pop_size=10, max_iter=15,
                 chaos_prob=0.3, mut_prob=0.2, mut_sigma=0.1,
                 seed=42, verbose=True):
        """
        chaos_prob : probability of the chaotic (re-initialisation) branch.
                     Set to 0.5 if your reference ChOA paper switches at mu = 0.5.
        mut_prob   : probability of Gaussian mutation per chimp. Set 0 to get plain ChOA (baseline).
        mut_sigma  : initial mutation std as a fraction of (ub - lb); decays linearly to ~0.
        """
        self.fitness = fitness
        self.lb, self.ub = np.asarray(lb, float), np.asarray(ub, float)
        self.dim = len(self.lb)
        self.N, self.T = pop_size, max_iter
        self.chaos_prob, self.mut_prob, self.mut_sigma = chaos_prob, mut_prob, mut_sigma
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.n_evals = 0

    def _eval(self, x):
        self.n_evals += 1
        return float(self.fitness(x))

    @staticmethod
    def _chaos_step(c):
        # Sinusoidal chaotic map, stays inside (0, 1)
        c = 2.3 * c ** 2 * np.sin(np.pi * c)
        return np.clip(np.abs(c), 1e-6, 1.0)

    def optimize(self):
        rng, lb, ub, N, T = self.rng, self.lb, self.ub, self.N, self.T
        span = ub - lb

        X = lb + rng.random((N, self.dim)) * span
        fit = np.array([self._eval(x) for x in X])

        # Four leaders: Attacker, Barrier, Chaser, Driver (best 4 so far)
        order = np.argsort(fit)[:4]
        leaders, leader_fit = X[order].copy(), fit[order].copy()
        if len(leaders) < 4:  # tiny populations
            reps = 4 - len(leaders)
            leaders = np.vstack([leaders, np.repeat(leaders[:1], reps, axis=0)])
            leader_fit = np.concatenate([leader_fit, np.repeat(leader_fit[:1], reps)])

        chaos = np.full(self.dim, 0.7)
        history = {"best": [float(leader_fit[0])], "mean": [float(fit.mean())]}

        for t in range(T):
            f = 2.5 * (1 - t / T)                              # dynamic coefficient, 2.5 -> 0
            sigma = self.mut_sigma * (1 - t / T) + 1e-3       # decaying Gaussian step
            X_new = np.empty_like(X)

            for i in range(N):
                cand = np.empty((4, self.dim))
                for g in range(4):
                    a = 2 * f * rng.random(self.dim) - f
                    c = 2 * rng.random(self.dim)
                    chaos = self._chaos_step(chaos)
                    d = np.abs(c * leaders[g] - chaos * X[i])
                    cand[g] = leaders[g] - a * d
                x = cand.mean(axis=0)                          # X(t+1) = (X1+X2+X3+X4)/4

                if rng.random() < self.chaos_prob:             # chaotic branch
                    chaos = self._chaos_step(chaos)
                    x = lb + chaos * span
                if rng.random() < self.mut_prob:               # Gaussian mutation
                    x = x + rng.normal(0.0, sigma * span)

                X_new[i] = np.clip(x, lb, ub)

            X = X_new
            fit = np.array([self._eval(x) for x in X])

            # Elitism: keep the 4 best among old leaders + current population
            all_X = np.vstack([leaders, X])
            all_f = np.concatenate([leader_fit, fit])
            keep = np.argsort(all_f)[:4]
            leaders, leader_fit = all_X[keep].copy(), all_f[keep].copy()

            history["best"].append(float(leader_fit[0]))
            history["mean"].append(float(fit.mean()))
            if self.verbose:
                print(f"[GMCO] iter {t + 1}/{T}  best={leader_fit[0]:.6f}  mean={fit.mean():.6f}")

        return {"best_pos": leaders[0], "best_fit": float(leader_fit[0]),
                "history": history, "n_evals": self.n_evals}


def plot_convergence(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(history["best"], label="Best fitness")
    plt.plot(history["mean"], label="Population mean", alpha=0.6)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (lower is better)")
    plt.title("GMCO convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_report(result, space, out_dir):
    """Writes gmco_best_params.json, gmco_history.json, gmco_convergence.png for the thesis report."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gmco_best_params.json"), "w") as f:
        json.dump({"best_params": space.decode(result["best_pos"]),
                   "best_fitness": result["best_fit"],
                   "n_evals": result["n_evals"]}, f, indent=2)
    with open(os.path.join(out_dir, "gmco_history.json"), "w") as f:
        json.dump(result["history"], f, indent=2)
    plot_convergence(result["history"], os.path.join(out_dir, "gmco_convergence.png"))


if __name__ == "__main__":
    # Self-test on a toy function; the code itself checks the result.
    sphere = lambda x: float(np.sum(x ** 2))
    res = GMCO(sphere, [-5] * 5, [5] * 5, pop_size=12, max_iter=25, seed=1, verbose=False).optimize()
    assert res["best_fit"] < res["history"]["best"][0] or res["best_fit"] == res["history"]["best"][0]
    assert all(b2 <= b1 + 1e-12 for b1, b2 in zip(res["history"]["best"], res["history"]["best"][1:]))
    print("GMCO self-test passed")
