# ============================================================
#  pso.py  –  Parçacık Sürü Optimizasyonu (PSO)
#             Binary wrapper tabanlı özellik seçimi
# ============================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import PSO, MODEL, RANDOM_SEED


class Particle:
    """Tek bir parçacığı temsil eder."""

    def __init__(self, n_features: int, rng: np.random.Generator):
        self.position  = rng.uniform(0.0, 1.0, n_features)   # [0,1] uzayında
        self.velocity  = rng.uniform(-0.5, 0.5, n_features)
        self.best_pos  = self.position.copy()
        self.best_fit  = -np.inf

    @property
    def selected(self) -> np.ndarray:
        """Eşik üzerindeki boyutlar seçili özelliği gösterir."""
        return self.position > PSO["threshold"]


class PSOFeatureSelector:
    """
    PSO tabanlı özellik seçici.

    Fitness fonksiyonu:
        f = accuracy(model, X_test[:, selected]) − penalty * (k / d)

    Burada k = seçilen özellik sayısı, d = toplam özellik sayısı.
    """

    def __init__(self):
        cfg = PSO
        self.n_particles  = cfg["n_particles"]
        self.n_iterations = cfg["n_iterations"]
        self.w            = cfg["w"]
        self.c1           = cfg["c1"]
        self.c2           = cfg["c2"]
        self.vel_clamp    = cfg["vel_clamp"]
        self.penalty      = cfg["penalty"]

        # İzleme
        self.history_best : list[float] = []
        self.history_avg  : list[float] = []
        self.best_mask    : np.ndarray | None = None

    # ── Fitness ──────────────────────────────────────────────
    def _fitness(
        self,
        position: np.ndarray,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_te: np.ndarray, y_te: np.ndarray,
    ) -> float:
        selected = position > PSO["threshold"]
        if selected.sum() == 0:
            return 0.0

        model = LogisticRegression(**MODEL)
        model.fit(X_tr[:, selected], y_tr)
        acc = accuracy_score(y_te, model.predict(X_te[:, selected]))

        ratio_penalty = self.penalty * (selected.sum() / len(position))
        return acc - ratio_penalty

    # ── Ana döngü ─────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test : np.ndarray, y_test : np.ndarray,
    ) -> np.ndarray:
        """
        PSO'yu çalıştırır ve en iyi özellik maskesini döndürür.

        Returns
        -------
        best_mask : bool ndarray  (n_features,)
        """
        rng       = np.random.default_rng(RANDOM_SEED)
        n         = X_train.shape[1]
        particles = [Particle(n, rng) for _ in range(self.n_particles)]

        # Başlangıç fitness
        for p in particles:
            p.best_fit = self._fitness(p.position, X_train, y_train, X_test, y_test)

        gbest_fit = max(p.best_fit for p in particles)
        gbest_pos = max(particles, key=lambda p: p.best_fit).best_pos.copy()

        print(f"\n{'İter':>5}  {'En İyi Fitness':>16}  {'Ort Fitness':>12}  {'Seçilen':>8}")
        print("─" * 50)

        for it in range(1, self.n_iterations + 1):
            fits = []
            for p in particles:
                r1 = rng.random(n)
                r2 = rng.random(n)

                # Hız güncelleme
                p.velocity = (
                    self.w  * p.velocity
                    + self.c1 * r1 * (p.best_pos  - p.position)
                    + self.c2 * r2 * (gbest_pos   - p.position)
                )
                p.velocity = np.clip(p.velocity, -self.vel_clamp, self.vel_clamp)

                # Pozisyon güncelleme
                p.position = np.clip(p.position + p.velocity, 0.0, 1.0)

                # Fitness
                fit = self._fitness(p.position, X_train, y_train, X_test, y_test)
                fits.append(fit)

                # Kişisel en iyi
                if fit > p.best_fit:
                    p.best_fit = fit
                    p.best_pos = p.position.copy()

                # Global en iyi
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest_pos = p.position.copy()

            self.history_best.append(gbest_fit)
            self.history_avg.append(float(np.mean(fits)))

            if it % 10 == 0:
                n_sel = int((gbest_pos > PSO["threshold"]).sum())
                print(f"{it:>5}  {gbest_fit:>16.4f}  {np.mean(fits):>12.4f}  {n_sel:>8}")

        print("─" * 50)
        self.best_mask = gbest_pos > PSO["threshold"]
        return self.best_mask
