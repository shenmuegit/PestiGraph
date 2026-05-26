"""
图谱数据 API：实体搜索、子图查询、实体详情
直接读 parquet 文件，供 Gradio UI 调用
"""
import json
from pathlib import Path

import pandas as pd

# 类型归一化：MiMo 提取的类型名称不统一，映射到标准类型
_TYPE_NORMALIZE = {
    "组织": "ORGANIZATION", "公司": "ORGANIZATION", "企业": "ORGANIZATION",
    "组织机构": "ORGANIZATION", "公司/组织": "ORGANIZATION", "组织/企业": "ORGANIZATION",
    "公司，组织": "ORGANIZATION",
    "农药产品": "PRODUCT", "产品": "PRODUCT", "农药": "PRODUCT",
    "农药有效成分": "CHEMICAL", "化学物质": "CHEMICAL", "化学成分": "CHEMICAL",
    "化学类别": "CHEMICAL", "化学分类": "CHEMICAL",
    "作物": "CROP", "作物/场所": "CROP", "作物场所": "CROP",
    "敏感作物": "CROP", "作物品种": "CROP", "作物类别": "CROP",
    "害虫": "PEST", "防治对象": "PEST", "害虫/防治对象": "PEST",
    "害虫类型": "PEST", "病害": "PEST", "病菌": "PEST", "杂草": "PEST",
    "杂草类型": "PEST", "杂草类别": "PEST", "病虫害": "PEST",
    "标识符": "ID", "标识符/编号": "ID", "标识符，ID": "ID",
    "登记证号": "ID", "标识符，农药登记证号": "ID",
    "时间点": "DATE", "日期": "DATE", "核准日期": "DATE",
    "重新核准日期": "DATE", "时间": "DATE", "时间点，DATE": "DATE",
    "概念": "CONCEPT", "使用限制": "CONCEPT", "用途": "CONCEPT",
    "剂型": "FORMULATION", "制剂类型": "FORMULATION",
    "施用方式": "METHOD", "施用方法": "METHOD", "使用方法": "METHOD",
}


def _normalize_type(t: str) -> str:
    """归一化实体类型"""
    if not t:
        return "OTHER"
    t_stripped = t.strip()
    # 直接匹配
    if t_stripped in _TYPE_NORMALIZE:
        return _TYPE_NORMALIZE[t_stripped]
    # 已经是标准类型
    if t_stripped.upper() in ("PRODUCT", "ORGANIZATION", "CHEMICAL", "CROP", "PEST",
                              "ID", "DATE", "CONCEPT", "FORMULATION", "METHOD"):
        return t_stripped.upper()
    # 模糊匹配：包含关键词
    for keyword, std in [("产品", "PRODUCT"), ("组织", "ORGANIZATION"), ("公司", "ORGANIZATION"),
                         ("化学", "CHEMICAL"), ("成分", "CHEMICAL"), ("作物", "CROP"),
                         ("害虫", "PEST"), ("病", "PEST"), ("虫", "PEST"), ("草", "PEST"),
                         ("标识", "ID"), ("证号", "ID"), ("LICENSE", "ID"),
                         ("时间", "DATE"), ("日期", "DATE")]:
        if keyword in t_stripped:
            return std
    return "OTHER"


_entities_df: pd.DataFrame | None = None
_relationships_df: pd.DataFrame | None = None


def _get_data_dir() -> Path:
    """获取数据目录"""
    return Path(__file__).resolve().parent / "data" / "output"


def _load():
    """懒加载 parquet 数据"""
    global _entities_df, _relationships_df
    if _entities_df is not None:
        return

    data_dir = _get_data_dir()
    _entities_df = pd.read_parquet(data_dir / "entities.parquet")
    _entities_df["type"] = _entities_df["type"].apply(_normalize_type)
    _relationships_df = pd.read_parquet(data_dir / "relationships.parquet")


def get_entity_types() -> list[str]:
    """获取所有实体类型"""
    _load()
    return sorted(_entities_df["type"].dropna().unique().tolist())


def search_entities(keyword: str, types: list[str] | None = None, limit: int = 50) -> list[dict]:
    """搜索实体"""
    _load()
    df = _entities_df

    if types:
        df = df[df["type"].isin(types)]

    if keyword.strip():
        mask = df["title"].str.contains(keyword, case=False, na=False)
        df = df[mask]

    df = df.nlargest(limit, "degree")

    return [
        {
            "title": row["title"],
            "type": str(row.get("type", "")),
            "description": str(row.get("description", "")),
            "degree": int(row.get("degree", 0)),
        }
        for _, row in df.iterrows()
    ]


def get_subgraph(entity_name: str, depth: int = 1) -> dict:
    """获取以指定实体为中心的子图"""
    _load()

    # 收集相关实体名
    visited = {entity_name}
    frontier = {entity_name}

    for _ in range(depth):
        new_frontier = set()
        for name in frontier:
            # 查找与该实体相关的所有关系
            rels = _relationships_df[
                (_relationships_df["source"] == name) | (_relationships_df["target"] == name)
            ]
            for _, row in rels.iterrows():
                new_frontier.add(row["source"])
                new_frontier.add(row["target"])
        frontier = new_frontier - visited
        visited |= frontier

    # 构建节点
    ent_df = _entities_df[_entities_df["title"].isin(visited)]
    title_to_id = {}
    nodes = []
    for _, row in ent_df.iterrows():
        nid = int(row["human_readable_id"])
        title_to_id[row["title"]] = nid
        nodes.append({
            "id": nid,
            "label": str(row["title"]),
            "type": str(row.get("type", "")),
            "description": str(row.get("description", "")),
            "degree": int(row.get("degree", 0)),
        })

    # 构建边
    rel_df = _relationships_df[
        _relationships_df["source"].isin(visited) & _relationships_df["target"].isin(visited)
    ]
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

    return {"nodes": nodes, "edges": edges}


def get_entity_detail(entity_name: str) -> dict | None:
    """获取实体详情和所有关系"""
    _load()

    ent = _entities_df[_entities_df["title"] == entity_name]
    if ent.empty:
        return None

    row = ent.iloc[0]
    rels = _relationships_df[
        (_relationships_df["source"] == entity_name) | (_relationships_df["target"] == entity_name)
    ]

    relations = []
    for _, r in rels.iterrows():
        other = r["target"] if r["source"] == entity_name else r["source"]
        relations.append({
            "entity": other,
            "description": str(r.get("description", "")),
            "weight": float(r.get("weight", 1.0)),
        })

    # 按权重排序
    relations.sort(key=lambda x: x["weight"], reverse=True)

    return {
        "title": str(row["title"]),
        "type": str(row.get("type", "")),
        "description": str(row.get("description", "")),
        "degree": int(row.get("degree", 0)),
        "relations": relations,
    }


def get_full_graph() -> dict:
    """获取全量图谱数据（排除孤立节点）"""
    _load()
    ent_df = _entities_df[_entities_df["degree"] > 0]
    rel_df = _relationships_df

    title_to_id = {}
    nodes = []
    for _, row in ent_df.iterrows():
        nid = int(row["human_readable_id"])
        title_to_id[row["title"]] = nid
        nodes.append({
            "id": nid,
            "label": str(row["title"]),
            "type": str(row.get("type", "")),
            "description": str(row.get("description", "")),
            "frequency": int(row.get("frequency", 0)),
            "degree": int(row.get("degree", 0)),
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

    return {"nodes": nodes, "edges": edges}


def get_products() -> list[str]:
    """获取所有农药产品名称，按关联度降序"""
    _load()
    df = _entities_df[_entities_df["type"] == "PRODUCT"]
    df = df.sort_values("degree", ascending=False)
    return df["title"].tolist()


def get_stats() -> dict:
    """获取数据统计"""
    _load()
    return {
        "entities": len(_entities_df),
        "relationships": len(_relationships_df),
        "types": get_entity_types(),
    }
