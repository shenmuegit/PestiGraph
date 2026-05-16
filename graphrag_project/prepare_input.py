"""
将 scraper/data/labels/*.json 的 full_text 提取为 GraphRAG 输入文本
"""
import json
import glob
from pathlib import Path

LABELS_DIR = Path(__file__).resolve().parent.parent / "scraper" / "data" / "labels"
OUTPUT_DIR = Path(__file__).resolve().parent / "input"
OUTPUT_DIR.mkdir(exist_ok=True)

count = 0
for fp in sorted(glob.glob(str(LABELS_DIR / "*.json"))):
    with open(fp, "r", encoding="utf-8") as f:
        rec = json.load(f)
    reg_no = rec.get("registration_no", Path(fp).stem)
    text = rec.get("full_text", "")
    if not text.strip():
        print(f"  [WARN] 跳过空文档: {fp}")
        continue
    out_path = OUTPUT_DIR / f"{reg_no}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    count += 1

print(f"已转换 {count} 个文件到 {OUTPUT_DIR}")
