"""Mini project: ML-like digital verification risk scoring without external dependencies.

This version is intentionally pure standard-library Python so it can run in restricted
environments (no pip/network required).
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class VerificationRun:
    block_type: str
    changed_lines: int
    toggling_activity: float
    lint_warnings: int
    cdc_violations: int
    timing_slack_ns: float
    coverage_delta_pct: float
    engineer_exp_years: int
    verification_fail: int


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build_synthetic_data(samples: int = 1200, seed: int = 42) -> List[VerificationRun]:
    rng = random.Random(seed)
    blocks = ["ALU", "CacheCtrl", "DMA", "PHY", "Interconnect"]
    data: List[VerificationRun] = []

    for _ in range(samples):
        block_type = rng.choice(blocks)
        changed_lines = rng.randint(5, 1200)
        toggling_activity = rng.uniform(0.05, 0.98)
        lint_warnings = max(0, int(rng.gauss(2.2, 1.4)))
        cdc_violations = max(0, int(rng.gauss(1.1, 1.0)))
        timing_slack_ns = rng.gauss(0.18, 0.35)
        coverage_delta_pct = rng.gauss(0.0, 3.8)
        engineer_exp_years = rng.randint(0, 15)

        logit = (
            0.0027 * changed_lines
            + 1.4 * toggling_activity
            + 0.85 * lint_warnings
            + 1.25 * cdc_violations
            - 2.0 * timing_slack_ns
            - 0.14 * coverage_delta_pct
            - 0.035 * engineer_exp_years
        )
        if block_type == "PHY":
            logit += 0.85
        elif block_type == "Interconnect":
            logit += 0.4

        p_fail = min(max(sigmoid(logit - 2.6), 0.01), 0.99)
        verification_fail = 1 if rng.random() < p_fail else 0

        data.append(
            VerificationRun(
                block_type,
                changed_lines,
                toggling_activity,
                lint_warnings,
                cdc_violations,
                timing_slack_ns,
                coverage_delta_pct,
                engineer_exp_years,
                verification_fail,
            )
        )
    return data


def risk_score(row: VerificationRun) -> float:
    block_bias = {"PHY": 0.85, "Interconnect": 0.4}.get(row.block_type, 0.0)
    z = (
        0.0025 * row.changed_lines
        + 1.35 * row.toggling_activity
        + 0.8 * row.lint_warnings
        + 1.2 * row.cdc_violations
        - 1.9 * row.timing_slack_ns
        - 0.12 * row.coverage_delta_pct
        - 0.03 * row.engineer_exp_years
        + block_bias
        - 2.45
    )
    return sigmoid(z)


def split_data(data: List[VerificationRun], test_ratio: float = 0.25, seed: int = 7) -> Tuple[List[VerificationRun], List[VerificationRun]]:
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def evaluate(test_data: List[VerificationRun], threshold: float = 0.5) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    probs = []
    labels = []

    for row in test_data:
        prob = risk_score(row)
        pred = 1 if prob >= threshold else 0
        actual = row.verification_fail
        probs.append(prob)
        labels.append(actual)

        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 0 and actual == 0:
            tn += 1
        elif pred == 1 and actual == 0:
            fp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(test_data) if test_data else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_predicted_risk": statistics.mean(probs) if probs else 0.0,
        "actual_fail_rate": statistics.mean(labels) if labels else 0.0,
    }


def print_top_risk_cases(rows: List[VerificationRun], top_n: int = 5) -> None:
    ranked = sorted(rows, key=risk_score, reverse=True)
    print("\n=== Highest-Risk Verification Runs ===")
    for idx, row in enumerate(ranked[:top_n], start=1):
        print(
            f"{idx}. block={row.block_type:<12} risk={risk_score(row):.3f} "
            f"actual_fail={row.verification_fail} changed_lines={row.changed_lines} "
            f"lint={row.lint_warnings} cdc={row.cdc_violations} slack={row.timing_slack_ns:.2f}ns"
        )


def export_csv(rows: List[VerificationRun], out_file: str = "verification_risk_report.csv") -> None:
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "block_type",
            "changed_lines",
            "toggling_activity",
            "lint_warnings",
            "cdc_violations",
            "timing_slack_ns",
            "coverage_delta_pct",
            "engineer_exp_years",
            "actual_fail",
            "predicted_fail_probability",
        ])
        for row in sorted(rows, key=risk_score, reverse=True):
            writer.writerow([
                row.block_type,
                row.changed_lines,
                f"{row.toggling_activity:.4f}",
                row.lint_warnings,
                row.cdc_violations,
                f"{row.timing_slack_ns:.4f}",
                f"{row.coverage_delta_pct:.4f}",
                row.engineer_exp_years,
                row.verification_fail,
                f"{risk_score(row):.6f}",
            ])


def main() -> None:
    data = build_synthetic_data()
    train_data, test_data = split_data(data)
    metrics = evaluate(test_data)

    print("=== Digital Verification Risk Model (No External Dependencies) ===")
    print(f"Train size: {len(train_data)} | Test size: {len(test_data)}")
    print("Confusion Matrix")
    print(f"TP={int(metrics['tp'])} FP={int(metrics['fp'])}")
    print(f"FN={int(metrics['fn'])} TN={int(metrics['tn'])}")
    print(
        f"Accuracy={metrics['accuracy']:.3f} Precision={metrics['precision']:.3f} "
        f"Recall={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
    )
    print(
        f"Average predicted risk={metrics['avg_predicted_risk']:.3f} | "
        f"Actual fail rate={metrics['actual_fail_rate']:.3f}"
    )

    print_top_risk_cases(test_data, top_n=5)
    export_csv(test_data)
    print("\nWrote: verification_risk_report.csv")


if __name__ == "__main__":
    main()
