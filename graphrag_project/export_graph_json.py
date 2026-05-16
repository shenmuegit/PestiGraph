"""
将 GraphRAG 输出的 entities.parquet / relationships.parquet 导出为 vis.js 可用的 JSON
用法: python export_graph_json.py [--top N]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=0,
                        help="Only export top N entities by node_degree (0 = all)")
    args = parser.parse_args()

    entities_path = OUTPUT_DIR / "entities.parquet"
    relationships_path = OUTPUT_DIR / "relationships.parquet"

    if not entities_path.exists():
        print(f"[ERROR] {entities_path} not found. Run graphrag index first.")
        return
    if not relationships_path.exists():
        print(f"[ERROR] {relationships_path} not found. Run graphrag index first.")
        return

    ent_df = pd.read_parquet(entities_path)
    rel_df = pd.read_parquet(relationships_path)

    print(f"Loaded {len(ent_df)} entities, {len(rel_df)} relationships")

    if args.top > 0:
        ent_df = ent_df.nlargest(args.top, "node_degree")
        titles = set(ent_df["title"])
        rel_df = rel_df[rel_df["source"].isin(titles) & rel_df["target"].isin(titles)]
        print(f"Filtered to top {args.top}: {len(ent_df)} entities, {len(rel_df)} relationships")

    title_to_id = {row["title"]: int(row["human_readable_id"]) for _, row in ent_df.iterrows()}

    nodes = []
    for _, row in ent_df.iterrows():
        nodes.append({
            "id": int(row["human_readable_id"]),
            "label": str(row["title"]),
            "type": str(row.get("type", "")),
            "description": str(row.get("description", "")),
            "frequency": int(row.get("node_frequency", 0)),
            "degree": int(row.get("node_degree", 0)),
        })

    edges = []
    for _, row in rel_df.iterrows():
        src_id = title_to_id.get(row["source"])
        tgt_id = title_to_id.get(row["target"])
        if src_id is None or tgt_id is None:
            continue
        edges.append({
            "from": src_id,
            "to": tgt_id,
            "source_label": str(row["source"]),
            "target_label": str(row["target"]),
            "label": str(row.get("description", "")),
            "weight": float(row.get("weight", 1.0)),
        })

    data = {"nodes": nodes, "edges": edges}
    out_path = OUTPUT_DIR / "graph_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Exported to {out_path} ({len(nodes)} nodes, {len(edges)} edges)")


if __name__ == "__main__":
    main()
