#!/usr/bin/env python3
"""Generate benchmark plots from benchmark-results.json."""

import json
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    # Filter out errors entirely.
    return [r for r in data if not r.get("error")]


def plot_detection_by_language(results, output="benchmark_by_language.png"):
    langs = sorted(set(r["language"] for r in results))
    detected = []
    line_overlap = []
    cwe_match = []
    totals = []

    for lang in langs:
        lr = [r for r in results if r["language"] == lang]
        totals.append(len(lr))
        detected.append(sum(1 for r in lr if r["detected"]))
        line_overlap.append(sum(1 for r in lr if r["line_overlap"]))
        cwe_match.append(sum(1 for r in lr if r["correct_cwe"]))

    x = range(len(langs))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar([i - width for i in x], [d / t * 100 for d, t in zip(detected, totals)],
                   width, label="File detected", color="#2ecc71")
    bars2 = ax.bar(list(x), [d / t * 100 for d, t in zip(line_overlap, totals)],
                   width, label="Line overlap", color="#3498db")
    bars3 = ax.bar([i + width for i in x], [d / t * 100 for d, t in zip(cwe_match, totals)],
                   width, label="CWE match", color="#9b59b6")

    ax.set_ylabel("Rate (%)")
    ax.set_title("Detection Accuracy by Language")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l}\n(n={t})" for l, t in zip(langs, totals)])
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels.
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def plot_detection_by_vuln_type(results, output="benchmark_by_vuln_type.png"):
    vtypes = sorted(set(r["vuln_type"] for r in results))
    detected = []
    line_overlap = []
    cwe_match = []
    totals = []

    for vt in vtypes:
        vr = [r for r in results if r["vuln_type"] == vt]
        totals.append(len(vr))
        detected.append(sum(1 for r in vr if r["detected"]))
        line_overlap.append(sum(1 for r in vr if r["line_overlap"]))
        cwe_match.append(sum(1 for r in vr if r["correct_cwe"]))

    x = range(len(vtypes))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], [d / t * 100 for d, t in zip(detected, totals)],
           width, label="File detected", color="#2ecc71")
    ax.bar(list(x), [d / t * 100 for d, t in zip(line_overlap, totals)],
           width, label="Line overlap", color="#3498db")
    ax.bar([i + width for i in x], [d / t * 100 for d, t in zip(cwe_match, totals)],
           width, label="CWE match", color="#9b59b6")

    ax.set_ylabel("Rate (%)")
    ax.set_title("Detection Accuracy by Vulnerability Type")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{vt}\n(n={t})" for vt, t in zip(vtypes, totals)])
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def plot_overall_summary(results, output="benchmark_overall.png"):
    total = len(results)
    detected = sum(1 for r in results if r["detected"])
    line_overlap = sum(1 for r in results if r["line_overlap"])
    cwe_match = sum(1 for r in results if r["correct_cwe"])

    categories = ["File\nDetected", "Line\nOverlap", "CWE\nMatch"]
    values = [detected / total * 100, line_overlap / total * 100, cwe_match / total * 100]
    counts = [f"{detected}/{total}", f"{line_overlap}/{total}", f"{cwe_match}/{total}"]
    colors = ["#2ecc71", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    for bar, count, val in zip(bars, counts, values):
        ax.annotate(f"{val:.1f}%\n({count})",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=12, fontweight="bold")

    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Overall Benchmark Results (n={total} cases, 1 hypothesis each)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def plot_timing(results, output="benchmark_timing.png"):
    # By language.
    langs = sorted(set(r["language"] for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot by language.
    ax = axes[0]
    lang_times = [[r["elapsed_seconds"] for r in results if r["language"] == l] for l in langs]
    bp = ax.boxplot(lang_times, labels=langs, patch_artist=True)
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors[:len(langs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Seconds")
    ax.set_title("Analysis Time by Language")
    ax.grid(axis="y", alpha=0.3)

    # Histogram of all times.
    ax = axes[1]
    times = [r["elapsed_seconds"] for r in results]
    ax.hist(times, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(sum(times) / len(times), color="#e74c3c", linestyle="--",
               label=f"Mean: {sum(times)/len(times):.0f}s")
    median = sorted(times)[len(times) // 2]
    ax.axvline(median, color="#2ecc71", linestyle="--",
               label=f"Median: {median:.0f}s")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Count")
    ax.set_title(f"Analysis Time Distribution (n={len(times)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark-results.json"
    results = load_results(path)
    print(f"Loaded {len(results)} evaluated results (errors excluded)")

    plot_overall_summary(results)
    plot_detection_by_language(results)
    plot_detection_by_vuln_type(results)
    plot_timing(results)


if __name__ == "__main__":
    main()
