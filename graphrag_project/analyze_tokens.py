"""
统计 input/*.txt 文档的 token 数量分布，建议最优 chunk size
用法: python analyze_tokens.py
"""
import math
from pathlib import Path

import tiktoken

INPUT_DIR = Path(__file__).resolve().parent / "input"
ENCODING = "o200k_base"


def main():
    enc = tiktoken.get_encoding(ENCODING)
    files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"Encoding: {ENCODING}")
    print(f"Documents: {len(files)}\n")

    results = []
    for fp in files:
        text = fp.read_text(encoding="utf-8")
        n_tokens = len(enc.encode(text))
        results.append((fp.name, n_tokens))

    tokens = [r[1] for r in results]
    tokens.sort()

    avg = sum(tokens) / len(tokens)
    median = tokens[len(tokens) // 2]
    max_val = max(tokens)
    min_val = min(tokens)

    print(f"Min:    {min_val} tokens")
    print(f"Max:    {max_val} tokens")
    print(f"Avg:    {avg:.0f} tokens")
    print(f"Median: {median} tokens")

    p90 = tokens[int(len(tokens) * 0.9)]
    p95 = tokens[int(len(tokens) * 0.95)]
    p99 = tokens[int(len(tokens) * 0.99)]
    print(f"P90:    {p90} tokens")
    print(f"P95:    {p95} tokens")
    print(f"P99:    {p99} tokens")

    # histogram
    buckets = [0, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 3000, 5000, 10000, 999999]
    print(f"\n{'Range':>16}  {'Count':>6}  Bar")
    print("-" * 60)
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = sum(1 for t in tokens if lo <= t < hi)
        label = f"{lo}-{hi}" if hi < 999999 else f"{lo}+"
        bar = "#" * min(count // 10, 60)
        print(f"{label:>16}  {count:>6}  {bar}")

    # top 10
    results.sort(key=lambda x: -x[1])
    print(f"\nTop 10 longest documents:")
    for name, n in results[:10]:
        print(f"  {name:40s}  {n:>6} tokens")

    # count docs that would be split at current 1200
    split_count = sum(1 for t in tokens if t > 1200)
    print(f"\nDocs > 1200 tokens (currently split): {split_count}")

    suggested = math.ceil(max_val / 100) * 100
    print(f"\nSuggested chunk size: {suggested} (max {max_val} rounded up to nearest 100)")
    print(f"With this setting: 0 documents will be split, chunks = {len(files)}")


if __name__ == "__main__":
    main()
