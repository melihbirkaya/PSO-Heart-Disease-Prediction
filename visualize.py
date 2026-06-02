# ============================================================
#  visualize.py  –  Tüm grafikler
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from config import OUTPUT_DIR, FEATURE_NAMES

# ── Genel stil ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family"  : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"    : True,
    "grid.alpha"   : 0.3,
    "figure.dpi"   : 150,
})
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
GRAY   = "#9CA3AF"
LIGHT  = "#F3F4F6"


def _save(fig: plt.Figure, filename: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFİK] Kaydedildi → {path}")
    return path


# ── 1. Sınıf dağılımı ────────────────────────────────────────
def plot_class_distribution(y: np.ndarray) -> str:
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    labels = ["Hastalık Yok (0)", "Hastalık Var (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts, color=[BLUE, RED], alpha=0.85, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4, str(val),
                ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Şekil 1 – Sınıf Dağılımı", fontweight="bold")
    ax.set_ylabel("Örnek Sayısı")
    ax.set_ylim(0, max(counts) * 1.2)
    return _save(fig, "fig1_class_distribution.png")


# ── 2. PSO yakınsama ─────────────────────────────────────────
def plot_convergence(history_best: list, history_avg: list) -> str:
    iters = range(1, len(history_best) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, history_best, color=BLUE,  lw=2,   label="En İyi Fitness")
    ax.plot(iters, history_avg,  color=GREEN, lw=1.5, ls="--", alpha=0.8, label="Ortalama Fitness")
    ax.fill_between(iters, history_avg, history_best, alpha=0.08, color=BLUE)
    ax.set_title("Şekil 2 – PSO Yakınsama Eğrisi", fontweight="bold")
    ax.set_xlabel("İterasyon")
    ax.set_ylabel("Fitness Değeri")
    ax.legend()
    return _save(fig, "fig2_convergence.png")


# ── 3. Seçilen özellikler ────────────────────────────────────
def plot_feature_selection(mask: np.ndarray) -> str:
    colors = [GREEN if s else GRAY for s in mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(FEATURE_NAMES, [1] * len(FEATURE_NAMES),
                   color=colors, edgecolor="white", alpha=0.9)
    for i, sel in enumerate(mask):
        ax.text(1.03, i, "✓ Seçildi" if sel else "✗ Elendi",
                va="center", fontsize=9,
                color=GREEN if sel else GRAY, fontweight="bold" if sel else "normal")
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.set_title("Şekil 3 – PSO Özellik Seçimi Sonucu", fontweight="bold")
    ax.invert_yaxis()
    return _save(fig, "fig3_feature_selection.png")


# ── 4–5. Karışıklık matrisleri ───────────────────────────────
def plot_confusion_matrices(cm_base: np.ndarray, cm_pso: np.ndarray,
                             acc_base: float, acc_pso: float) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    tick_labels = ["Yok", "Var"]

    for ax, cm, title, cmap, acc in zip(
        axes,
        [cm_base, cm_pso],
        [f"Şekil 4 – Baseline\nDoğruluk: %{acc_base*100:.2f}",
         f"Şekil 5 – PSO\nDoğruluk: %{acc_pso*100:.2f}"],
        ["Blues", "Greens"],
        [acc_base, acc_pso],
    ):
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Gerçek Etiket")
        ax.set_xlabel("Tahmin Edilen")

    fig.suptitle("Karışıklık Matrisi Karşılaştırması", fontweight="bold", y=1.02)
    return _save(fig, "fig4_confusion_matrices.png")


# ── 6. Performans çubuk grafiği ──────────────────────────────
def plot_performance_comparison(results_base: dict, results_pso: dict) -> str:
    metrics     = ["Test Doğruluğu", "ROC-AUC", f"{5}-Fold CV"]
    base_vals   = [results_base["accuracy"], results_base["roc_auc"], results_base["cv_mean"]]
    pso_vals    = [results_pso ["accuracy"], results_pso ["roc_auc"], results_pso ["cv_mean"]]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, [v * 100 for v in base_vals], w,
                label=f"Baseline ({results_base['n_features']} özellik)",
                color=BLUE, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, [v * 100 for v in pso_vals], w,
                label=f"PSO ({results_pso['n_features']} özellik)",
                color=GREEN, alpha=0.85, edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Şekil 6 – Baseline vs PSO Performans Karşılaştırması", fontweight="bold")
    ax.set_ylabel("Skor (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(60, 105)
    ax.legend()
    return _save(fig, "fig6_performance_comparison.png")


# ── 7. Tüm grafikleri tek sayfada birleştir ──────────────────
def plot_all_in_one(
    y            : np.ndarray,
    history_best : list,
    history_avg  : list,
    mask         : np.ndarray,
    cm_base      : np.ndarray,
    cm_pso       : np.ndarray,
    results_base : dict,
    results_pso  : dict,
) -> str:
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # 1 – Sınıf dağılımı
    ax1 = fig.add_subplot(gs[0, 0])
    counts = [(y == 0).sum(), (y == 1).sum()]
    bars = ax1.bar(["Yok (0)", "Var (1)"], counts, color=[BLUE, RED], alpha=0.85, edgecolor="white")
    for b, v in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height()+3, str(v), ha="center", fontweight="bold")
    ax1.set_title("Şekil 1 – Sınıf Dağılımı", fontweight="bold")
    ax1.set_ylabel("Örnek Sayısı")
    ax1.set_ylim(0, max(counts) * 1.2)

    # 2 – Yakınsama
    ax2 = fig.add_subplot(gs[0, 1])
    iters = range(1, len(history_best)+1)
    ax2.plot(iters, history_best, color=BLUE,  lw=2, label="En İyi")
    ax2.plot(iters, history_avg,  color=GREEN, lw=1.5, ls="--", alpha=0.8, label="Ortalama")
    ax2.fill_between(iters, history_avg, history_best, alpha=0.08, color=BLUE)
    ax2.set_title("Şekil 2 – PSO Yakınsama", fontweight="bold")
    ax2.set_xlabel("İterasyon"); ax2.set_ylabel("Fitness")
    ax2.legend(fontsize=8)

    # 3 – Özellik seçimi
    ax3 = fig.add_subplot(gs[0, 2])
    colors = [GREEN if s else GRAY for s in mask]
    ax3.barh(FEATURE_NAMES, [1]*len(FEATURE_NAMES), color=colors, edgecolor="white", alpha=0.9)
    for i, sel in enumerate(mask):
        ax3.text(1.03, i, "✓" if sel else "✗", va="center", fontsize=9,
                 color=GREEN if sel else GRAY, fontweight="bold")
    ax3.set_xlim(0, 1.4); ax3.set_xticks([])
    ax3.set_title("Şekil 3 – Seçilen Özellikler", fontweight="bold")
    ax3.invert_yaxis()

    # 4 – Karışıklık: Baseline
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=ax4,
                xticklabels=["Yok","Var"], yticklabels=["Yok","Var"],
                linewidths=0.5, linecolor="white", annot_kws={"size":12,"weight":"bold"})
    ax4.set_title(f"Şekil 4 – Baseline Karışıklık\n%{results_base['accuracy']*100:.2f}", fontweight="bold")
    ax4.set_ylabel("Gerçek"); ax4.set_xlabel("Tahmin")

    # 5 – Karışıklık: PSO
    ax5 = fig.add_subplot(gs[1, 1])
    sns.heatmap(cm_pso, annot=True, fmt="d", cmap="Greens", ax=ax5,
                xticklabels=["Yok","Var"], yticklabels=["Yok","Var"],
                linewidths=0.5, linecolor="white", annot_kws={"size":12,"weight":"bold"})
    ax5.set_title(f"Şekil 5 – PSO Karışıklık\n%{results_pso['accuracy']*100:.2f}", fontweight="bold")
    ax5.set_ylabel("Gerçek"); ax5.set_xlabel("Tahmin")

    # 6 – Performans karşılaştırması
    ax6 = fig.add_subplot(gs[1, 2])
    metrics   = ["Test Doğr.", "ROC-AUC", "5-Fold CV"]
    base_vals = [results_base["accuracy"], results_base["roc_auc"], results_base["cv_mean"]]
    pso_vals  = [results_pso ["accuracy"], results_pso ["roc_auc"], results_pso ["cv_mean"]]
    x = np.arange(len(metrics)); w = 0.35
    b1 = ax6.bar(x-w/2, [v*100 for v in base_vals], w, label=f"Baseline ({results_base['n_features']} öz.)", color=BLUE,  alpha=0.85, edgecolor="white")
    b2 = ax6.bar(x+w/2, [v*100 for v in pso_vals ], w, label=f"PSO ({results_pso['n_features']} öz.)",      color=GREEN, alpha=0.85, edgecolor="white")
    for bars in [b1, b2]:
        for bar in bars:
            ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax6.set_title("Şekil 6 – Performans Karşılaştırması", fontweight="bold")
    ax6.set_xticks(x); ax6.set_xticklabels(metrics, fontsize=9)
    ax6.set_ylabel("Skor (%)"); ax6.set_ylim(60, 108); ax6.legend(fontsize=8)

    fig.suptitle("PSO Tabanlı Özellik Seçimi – Kalp Hastalığı Tahmini",
                 fontsize=14, fontweight="bold", y=1.01)

    return _save(fig, "fig_all.png")
