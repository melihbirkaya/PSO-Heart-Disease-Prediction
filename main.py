# ============================================================
#  main.py  –  Ana çalıştırma dosyası
#
#  Kullanım:
#      cd src
#      python main.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from data_loader import load_data, preprocess
from pso         import PSOFeatureSelector
from evaluate    import train_and_evaluate
from visualize   import (
    plot_class_distribution,
    plot_convergence,
    plot_feature_selection,
    plot_confusion_matrices,
    plot_performance_comparison,
    plot_all_in_one,
)
from config import FEATURE_NAMES, OUTPUT_DIR


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Veri ──────────────────────────────────────────────
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess(X, y)

    # Tam ölçeklenmiş X (CV için)
    from sklearn.preprocessing import StandardScaler
    X_full = StandardScaler().fit_transform(X)

    # ── 2. Baseline ──────────────────────────────────────────
    results_base = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        X_full, y,
        mask=None, label="Baseline (Tüm Özellikler)",
    )

    # ── 3. PSO ───────────────────────────────────────────────
    pso  = PSOFeatureSelector()
    mask = pso.fit(X_train, y_train, X_test, y_test)

    selected = [feature_names[i] for i, s in enumerate(mask) if s]
    print(f"\n[PSO] Seçilen {mask.sum()} özellik: {selected}")

    results_pso = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        X_full, y,
        mask=mask, label=f"PSO ({mask.sum()} Özellik)",
    )

    # ── 4. Karşılaştırma özeti ────────────────────────────────
    print("\n" + "═"*55)
    print("  ÖZET KARŞILAŞTIRMA")
    print("═"*55)
    print(f"  {'Metrik':<22} {'Baseline':>12} {'PSO':>12}")
    print("─"*55)
    print(f"  {'Özellik Sayısı':<22} {results_base['n_features']:>12} {results_pso['n_features']:>12}")
    print(f"  {'Test Doğruluğu':<22} {results_base['accuracy']*100:>11.2f}% {results_pso['accuracy']*100:>11.2f}%")
    print(f"  {'ROC-AUC':<22} {results_base['roc_auc']:>12.4f} {results_pso['roc_auc']:>12.4f}")
    print(f"  {'5-Fold CV':<22} {results_base['cv_mean']*100:>11.2f}% {results_pso['cv_mean']*100:>11.2f}%")
    delta = (results_pso['accuracy'] - results_base['accuracy']) * 100
    print(f"  {'Net Değişim':<22} {'—':>12} {delta:>+11.2f}%")
    print("═"*55)

    # ── 5. Grafikler ─────────────────────────────────────────
    print("\n[GRAFİK] Oluşturuluyor...")
    plot_class_distribution(y)
    plot_convergence(pso.history_best, pso.history_avg)
    plot_feature_selection(mask)
    plot_confusion_matrices(
        results_base["confusion_matrix"],
        results_pso ["confusion_matrix"],
        results_base["accuracy"],
        results_pso ["accuracy"],
    )
    plot_performance_comparison(results_base, results_pso)
    plot_all_in_one(
        y, pso.history_best, pso.history_avg, mask,
        results_base["confusion_matrix"],
        results_pso ["confusion_matrix"],
        results_base, results_pso,
    )
    print(f"\n[TAMAMLANDI] Tüm çıktılar → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
