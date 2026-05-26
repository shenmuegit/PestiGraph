---
title: PestiGraph
emoji: "\U0001F33F"
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
---

# PestiGraph - 农药知识图谱智能问答系统

基于 Microsoft GraphRAG 构建的农药知识图谱，支持智能问答和交互式图谱可视化。

## 功能

**智能问答**
- 自动路由：根据问题类型自动选择 Global / Local / Drift / Basic 查询模式
- 可调参数：社区层级（community_level）和回答格式（response_type）
- 实时日志：查询过程中展示 LLM 调用阶段和耗时

**知识图谱浏览**
- 全量实体和关系的交互式可视化（vis.js）
- 按类型筛选：PRODUCT、CHEMICAL、ORGANIZATION、CROP、PEST 等 10 种类型
- 实体搜索、点击查看详情和关系

## 技术栈

- **GraphRAG** — 微软图谱增强检索生成框架
- **MiMo v2.5-pro** — 推理 LLM
- **Gradio** — Web UI
- **LanceDB** — 向量存储
- **vis.js** — 图谱可视化

## 数据

当前索引：200 篇农药登记标签文档，提取 1918 个实体、3335 条关系。

## 本地运行

```bash
cd app
pip install -r requirements.txt
# 配置环境变量
cp data/.env.example data/.env
# 编辑 data/.env 填入 API Key
python app.py
```

访问 http://localhost:7860

## 环境变量

| 变量 | 说明 |
|------|------|
| `MIMO_API_KEY` | MiMo LLM API Key |
| `DASHSCOPE_API_KEY` | 阿里云 DashScope Embedding Key |
