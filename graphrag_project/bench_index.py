"""
GraphRAG 索引基准测试脚本
独立于正式数据，所有测试数据存放在 bench/ 目录

子命令:
  index  - 索引测试（可切换文档数量和嵌入模型）
  query  - 交互式查询 / 知识图谱可视化

用法:
  python bench_index.py index --count 100 --embedding bge
  python bench_index.py index --count 500 --embedding v4
  python bench_index.py query
  python bench_index.py query --view [--top 200] [--port 8081]
"""
import argparse
import http.server
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parent
FULL_INPUT_DIR = ROOT / "input"
BENCH_DIR = ROOT / "bench"
BENCH_SETTINGS = BENCH_DIR / "settings.yaml"

EMBEDDING_PROFILES = {
    "bge": {
        "model_provider": "openai",
        "model": "bge-m3",
        "api_base": "http://localhost:11434/v1",
        "auth_method": "api_key",
        "api_key": "ollama",
        "retry": {"type": "exponential_backoff", "max_retries": 2},
    },
    "v4": {
        "model_provider": "openai",
        "model": "text-embedding-v4",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "auth_method": "api_key",
        "api_key": "${DASHSCOPE_API_KEY}",
        "call_args": {"encoding_format": "float"},
        "retry": {"type": "exponential_backoff", "max_retries": 2},
    },
}


# ─── index 子命令 ───────────────────────────────────────────────

def cmd_index(args):
    """索引测试数据"""
    # 1. 初始化 bench 目录
    init_bench_dir()

    # 2. 准备输入文件
    actual_count = prepare_input(args.count)

    # 3. 清理旧索引
    for d in ["output", "cache", "logs"]:
        p = BENCH_DIR / d
        if p.exists():
            shutil.rmtree(p)

    # 4. 生成 settings.yaml
    write_bench_settings(args.embedding)

    # 5. 运行索引
    print(f"\n开始索引 {actual_count} 个文档 (embedding: {args.embedding})...\n")
    cmd = [sys.executable, "-m", "graphrag", "index", "-r", str(BENCH_DIR)]
    start = time.time()
    result = subprocess.run(cmd, cwd=str(BENCH_DIR))
    elapsed = time.time() - start

    # 6. 输出报告
    report(actual_count, args.embedding, elapsed, result.returncode)


def init_bench_dir():
    """初始化 bench 目录结构"""
    BENCH_DIR.mkdir(exist_ok=True)
    (BENCH_DIR / "input").mkdir(exist_ok=True)

    # 复制 prompts
    src_prompts = ROOT / "prompts"
    dst_prompts = BENCH_DIR / "prompts"
    if dst_prompts.exists():
        shutil.rmtree(dst_prompts)
    shutil.copytree(src_prompts, dst_prompts)

    # 复制 .env
    src_env = ROOT / ".env"
    if src_env.exists():
        shutil.copy2(src_env, BENCH_DIR / ".env")


def prepare_input(count: int) -> int:
    """从全量 input 中随机抽取文件"""
    bench_input = BENCH_DIR / "input"
    if bench_input.exists():
        shutil.rmtree(bench_input)
    bench_input.mkdir()

    all_files = sorted(FULL_INPUT_DIR.glob("*.txt"))
    if not all_files:
        print(f"[ERROR] {FULL_INPUT_DIR} 中没有 txt 文件，请先运行 prepare_input.py")
        sys.exit(1)

    if count >= len(all_files):
        selected = all_files
    else:
        selected = random.sample(all_files, count)

    for f in selected:
        shutil.copy2(f, bench_input / f.name)

    print(f"已准备 {len(selected)} 个文档到 {bench_input}")
    return len(selected)


def write_bench_settings(embedding: str):
    """基于正式 settings.yaml 生成 bench 专用配置"""
    with open(ROOT / "settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 切换嵌入模型
    cfg["embedding_models"]["default_embedding_model"] = EMBEDDING_PROFILES[embedding]

    # 阿里云 text-embedding-v4 单次最多 10 个文本，batch_max_tokens=8191
    if embedding == "v4":
        cfg["embed_text"]["batch_size"] = 10
        cfg["embed_text"]["batch_max_tokens"] = 8191

    # 所有路径指向 bench 内部（相对路径）
    cfg["input_storage"]["base_dir"] = "input"
    cfg["output_storage"]["base_dir"] = "output"
    cfg["reporting"]["base_dir"] = "logs"
    cfg["cache"]["storage"]["base_dir"] = "cache"
    cfg["vector_store"]["db_uri"] = "output\\lancedb"

    with open(BENCH_SETTINGS, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"嵌入模型: {EMBEDDING_PROFILES[embedding]['model']}")


def report(count: int, embedding: str, elapsed: float, returncode: int):
    """输出测试结果"""
    status = "成功" if returncode == 0 else f"失败 (exit={returncode})"
    entities_count = relationships_count = communities_count = "N/A"
    try:
        bench_output = BENCH_DIR / "output"
        for name, var in [("entities.parquet", "e"), ("relationships.parquet", "r"), ("communities.parquet", "c")]:
            p = bench_output / name
            if p.exists():
                n = len(pd.read_parquet(p))
                if var == "e":
                    entities_count = n
                elif var == "r":
                    relationships_count = n
                else:
                    communities_count = n
    except Exception:
        pass

    minutes = elapsed / 60
    model_name = EMBEDDING_PROFILES[embedding]["model"]
    print("\n" + "=" * 60)
    print("  GraphRAG 索引基准测试结果")
    print("=" * 60)
    print(f"  文档数量:   {count}")
    print(f"  嵌入模型:   {model_name}")
    print(f"  状态:       {status}")
    print(f"  耗时:       {minutes:.1f} 分钟 ({elapsed:.0f} 秒)")
    print(f"  实体数:     {entities_count}")
    print(f"  关系数:     {relationships_count}")
    print(f"  社区数:     {communities_count}")
    if isinstance(entities_count, int) and elapsed > 0:
        print(f"  速度:       {count / minutes:.1f} 文档/分钟")
    print(f"  数据目录:   {BENCH_DIR}")
    print("=" * 60)
    print(f"\n后续操作:")
    print(f"  查询:  python bench_index.py query")
    print(f"  可视化: python bench_index.py query --view")


# ─── query 子命令 ───────────────────────────────────────────────

def cmd_query(args):
    """交互式查询，或 --view 启动可视化"""
    bench_output = BENCH_DIR / "output"
    if not (bench_output / "entities.parquet").exists():
        print("[ERROR] bench 索引不存在，请先运行: python bench_index.py index --count N --embedding MODEL")
        sys.exit(1)

    if args.view:
        _start_viewer(bench_output, args.top, args.port)
        return

    methods = {"l": "local", "g": "global", "d": "drift", "b": "basic"}

    print(f"""
╔══════════════════════════════════════════════════════╗
║        GraphRAG Bench 查询 (数据: bench/)           ║
╠══════════════════════════════════════════════════════╣
║  /l - Local   /g - Global   /d - Drift   /b - Basic║
║  /help - 帮助   /q - 退出                           ║
╚══════════════════════════════════════════════════════╝""")

    method = None
    while True:
        try:
            if method:
                prompt = f"[bench|{method}] > "
            else:
                prompt = "[bench|请选择 /l /g /d /b] > "
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question == "/q":
            print("再见！")
            break
        if question.startswith("/"):
            key = question[1:]
            if key in methods:
                method = methods[key]
                print(f"已切换到 {method} 模式")
            elif key == "help":
                print("  /l Local  /g Global  /d Drift  /b Basic  /q 退出")
            else:
                print(f"未知命令: {question}")
            continue
        if not method:
            print("请先选择查询模式：/l /g /d /b")
            continue

        print(f"\n正在查询（{method}）...\n")
        cmd = [
            sys.executable, "-m", "graphrag", "query",
            "-r", str(BENCH_DIR),
            "-m", method,
            question,
        ]
        try:
            result = subprocess.run(
                cmd, cwd=str(BENCH_DIR),
                capture_output=True, text=True, encoding="utf-8", timeout=300,
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"[ERROR] {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("[ERROR] 查询超时（300秒）")
        print()


# ─── 可视化 ────────────────────────────────────────────────────

def _start_viewer(output_dir: Path, top: int, port: int):
    """导出数据并启动知识图谱可视化"""
    export_graph_json(output_dir, top)
    _write_viewer_html(BENCH_DIR / "graph_viewer.html")

    os.chdir(str(BENCH_DIR))
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("0.0.0.0", port), handler)

    url = f"http://localhost:{port}/graph_viewer.html"
    print(f"\n=== Bench Viewer: {url} ===")
    print("Press Ctrl+C to stop.\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


def export_graph_json(output_dir: Path, top: int):
    """导出 parquet 为 vis.js JSON"""
    ent_df = pd.read_parquet(output_dir / "entities.parquet")
    rel_df = pd.read_parquet(output_dir / "relationships.parquet")
    print(f"Loaded {len(ent_df)} entities, {len(rel_df)} relationships")

    if top > 0:
        ent_df = ent_df.nlargest(top, "degree")
        titles = set(ent_df["title"])
        rel_df = rel_df[rel_df["source"].isin(titles) & rel_df["target"].isin(titles)]
        print(f"Filtered to top {top}: {len(ent_df)} entities, {len(rel_df)} relationships")

    title_to_id = {row["title"]: int(row["human_readable_id"]) for _, row in ent_df.iterrows()}

    nodes = []
    for _, row in ent_df.iterrows():
        nodes.append({
            "id": int(row["human_readable_id"]),
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

    data = {"nodes": nodes, "edges": edges}
    out_path = output_dir / "graph_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Exported {len(nodes)} nodes, {len(edges)} edges → {out_path}")


def _write_viewer_html(path: Path):
    """生成内嵌的知识图谱可视化 HTML"""
    html = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>GraphRAG Entity Viewer</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:"Microsoft YaHei",sans-serif; background:#0f0f1a; color:#ddd; display:flex; height:100vh; overflow:hidden; }
#left-panel {
  width: 280px; min-width: 280px; background: #161625; border-right: 1px solid #2a2a40;
  display: flex; flex-direction: column; overflow: hidden;
}
#left-panel h2 { padding: 16px; font-size: 16px; color: #fff; border-bottom: 1px solid #2a2a40; }
#search-box {
  margin: 10px 12px; padding: 8px 12px; background: #1e1e35; border: 1px solid #3a3a55;
  border-radius: 6px; color: #fff; font-size: 14px; outline: none;
}
#search-box:focus { border-color: #5a7aff; }
#search-box::placeholder { color: #666; }
#type-filters { padding: 8px 12px; border-bottom: 1px solid #2a2a40; }
.filter-item { display: flex; align-items: center; padding: 3px 0; cursor: pointer; font-size: 13px; }
.filter-item input { margin-right: 8px; }
.filter-dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; display: inline-block; }
.filter-count { margin-left: auto; color: #666; font-size: 12px; }
#entity-list { flex: 1; overflow-y: auto; padding: 8px 0; }
.entity-group-title {
  padding: 6px 12px; font-size: 12px; color: #888; text-transform: uppercase;
  position: sticky; top: 0; background: #161625;
}
.entity-item {
  padding: 5px 12px 5px 24px; font-size: 13px; cursor: pointer;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.entity-item:hover { background: #1e1e35; }
#graph-container { flex: 1; position: relative; }
#graph { width: 100%; height: 100%; }
#loading {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
  font-size: 18px; color: #888;
}
#stats {
  position: absolute; top: 12px; left: 12px; font-size: 12px; color: #666;
  background: rgba(0,0,0,0.5); padding: 6px 10px; border-radius: 4px;
}
#right-panel {
  width: 320px; min-width: 320px; background: #161625; border-left: 1px solid #2a2a40;
  display: flex; flex-direction: column; overflow: hidden;
}
#detail-header { padding: 16px; font-size: 16px; color: #fff; border-bottom: 1px solid #2a2a40; }
#detail-content { flex: 1; overflow-y: auto; padding: 12px 16px; }
.detail-section { margin-bottom: 16px; }
.detail-label { font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 4px; }
.detail-value { font-size: 14px; line-height: 1.5; }
.detail-type-badge {
  display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px;
  color: #fff; margin-top: 4px;
}
.rel-item {
  padding: 8px; margin: 4px 0; background: #1e1e35; border-radius: 6px;
  font-size: 13px; cursor: pointer; line-height: 1.4;
}
.rel-item:hover { background: #252545; }
.rel-arrow { color: #5a7aff; margin: 0 4px; }
.rel-desc { color: #999; font-size: 12px; display: block; margin-top: 2px; }
</style>
</head>
<body>
<div id="left-panel">
  <h2>Entity Explorer</h2>
  <input id="search-box" type="text" placeholder="Search entities...">
  <div id="type-filters"></div>
  <div id="entity-list"></div>
</div>
<div id="graph-container">
  <div id="graph"></div>
  <div id="loading">Loading graph data...</div>
  <div id="stats"></div>
</div>
<div id="right-panel">
  <div id="detail-header">Details</div>
  <div id="detail-content">
    <p style="color:#666; font-size:13px;">Click a node or edge to view details.</p>
  </div>
</div>
<script>
const TYPE_COLORS = {
  "PRODUCT":"#e74c3c","ORGANIZATION":"#3498db","CHEMICAL":"#2ecc71",
  "FORMULATION":"#f39c12","CROP":"#27ae60","PEST":"#e67e22",
  "ID":"#9b59b6","DATE":"#8e44ad","CONCEPT":"#1abc9c",
};
const DEFAULT_COLOR = "#7f8c8d";
function getColor(type) { return TYPE_COLORS[type] || TYPE_COLORS[type?.toUpperCase()] || DEFAULT_COLOR; }

let network=null, allNodes=null, allEdges=null, graphData=null, hiddenTypes=new Set();

async function init() {
  try {
    const resp = await fetch("output/graph_data.json");
    if (!resp.ok) throw new Error("HTTP "+resp.status);
    graphData = await resp.json();
  } catch(e) { document.getElementById("loading").textContent="Failed: "+e.message; return; }

  document.getElementById("loading").style.display="none";
  const degreeMax = Math.max(...graphData.nodes.map(n=>n.degree),1);
  const nodesArr = graphData.nodes.map(n=>({
    id:n.id, label:n.label, _type:n.type, _description:n.description,
    _frequency:n.frequency, _degree:n.degree,
    size: 8+Math.sqrt(n.degree/degreeMax)*30,
    color:{background:getColor(n.type),border:getColor(n.type),highlight:{background:"#fff",border:getColor(n.type)}},
    font:{color:"#eee",size:11,face:"Microsoft YaHei"}, borderWidth:1.5,
  }));
  const weightMax = Math.max(...graphData.edges.map(e=>e.weight),1);
  const edgesArr = graphData.edges.map((e,i)=>({
    id:i, from:e.from, to:e.to, _source_label:e.source_label, _target_label:e.target_label,
    _label:e.label, _weight:e.weight,
    width:0.5+(e.weight/weightMax)*3, arrows:"to",
    color:{color:"rgba(150,150,180,0.4)",highlight:"#5a7aff"}, smooth:{type:"continuous"},
  }));

  allNodes=new vis.DataSet(nodesArr); allEdges=new vis.DataSet(edgesArr);
  const container=document.getElementById("graph");
  network=new vis.Network(container,{nodes:allNodes,edges:allEdges},{
    physics:{solver:"barnesHut",barnesHut:{gravitationalConstant:-3000,centralGravity:0.2,springLength:120,springConstant:0.02},stabilization:{iterations:150,updateInterval:25}},
    interaction:{hover:true,tooltipDelay:200,zoomView:true,dragView:true,hideEdgesOnDrag:true},
    layout:{improvedLayout:false},
  });
  network.once("stabilizationIterationsDone",()=>{ network.setOptions({physics:{enabled:false}}); });
  network.on("click",onNetworkClick);
  document.getElementById("stats").textContent=graphData.nodes.length+" entities | "+graphData.edges.length+" relationships";
  buildTypeFilters(); buildEntityList(); setupSearch();
}

function buildTypeFilters() {
  const counts={}; graphData.nodes.forEach(n=>{counts[n.type]=(counts[n.type]||0)+1;});
  const container=document.getElementById("type-filters");
  Object.keys(counts).sort((a,b)=>counts[b]-counts[a]).forEach(type=>{
    const div=document.createElement("div"); div.className="filter-item";
    div.innerHTML='<input type="checkbox" checked data-type="'+type+'"><span class="filter-dot" style="background:'+getColor(type)+'"></span>'+type+'<span class="filter-count">'+counts[type]+'</span>';
    div.querySelector("input").addEventListener("change",onFilterChange);
    container.appendChild(div);
  });
}
function onFilterChange() {
  hiddenTypes.clear();
  document.querySelectorAll("#type-filters input").forEach(cb=>{if(!cb.checked)hiddenTypes.add(cb.dataset.type);});
  allNodes.forEach(node=>{allNodes.update({id:node.id,hidden:hiddenTypes.has(node._type)});});
  allEdges.forEach(edge=>{
    const f=allNodes.get(edge.from),t=allNodes.get(edge.to);
    allEdges.update({id:edge.id,hidden:!f||!t||f.hidden||t.hidden});
  });
}
function buildEntityList(filter) {
  const container=document.getElementById("entity-list"); container.innerHTML="";
  const grouped={}; graphData.nodes.forEach(n=>{
    if(filter&&!n.label.includes(filter))return;
    if(!grouped[n.type])grouped[n.type]=[]; grouped[n.type].push(n);
  });
  Object.keys(grouped).sort().forEach(type=>{
    const title=document.createElement("div"); title.className="entity-group-title";
    title.textContent=type+" ("+grouped[type].length+")"; container.appendChild(title);
    grouped[type].sort((a,b)=>b.degree-a.degree).forEach(n=>{
      const item=document.createElement("div"); item.className="entity-item";
      item.textContent=n.label; item.addEventListener("click",()=>focusNode(n.id));
      container.appendChild(item);
    });
  });
}
function setupSearch() {
  let timer; document.getElementById("search-box").addEventListener("input",e=>{
    clearTimeout(timer); timer=setTimeout(()=>{
      const q=e.target.value.trim(); buildEntityList(q||undefined);
      if(q){const m=graphData.nodes.find(n=>n.label.includes(q)); if(m)focusNode(m.id);}
    },300);
  });
}
function focusNode(nodeId) {
  network.focus(nodeId,{scale:1.5,animation:{duration:400,easingFunction:"easeInOutQuad"}});
  network.selectNodes([nodeId]); showNodeDetail(nodeId);
}
function onNetworkClick(params) {
  if(params.nodes.length>0) showNodeDetail(params.nodes[0]);
  else if(params.edges.length>0) showEdgeDetail(params.edges[0]);
}
function showNodeDetail(nodeId) {
  const node=allNodes.get(nodeId); if(!node)return;
  const rels=[]; allEdges.forEach(edge=>{
    if(edge.from===nodeId||edge.to===nodeId){
      const other=edge.from===nodeId?allNodes.get(edge.to):allNodes.get(edge.from);
      rels.push({edge,other,direction:edge.from===nodeId?"out":"in"});
    }
  });
  document.getElementById("detail-header").textContent=node.label;
  document.getElementById("detail-content").innerHTML=
    '<div class="detail-section"><div class="detail-label">Entity</div><div class="detail-value" style="font-size:18px;font-weight:bold;">'+node.label+'</div></div>'+
    '<div class="detail-section"><div class="detail-label">Type</div><span class="detail-type-badge" style="background:'+getColor(node._type)+'">'+node._type+'</span></div>'+
    '<div class="detail-section"><div class="detail-label">Description</div><div class="detail-value">'+(node._description||"N/A")+'</div></div>'+
    '<div class="detail-section"><div class="detail-label">Stats</div><div class="detail-value">Degree: '+node._degree+' | Frequency: '+node._frequency+'</div></div>'+
    '<div class="detail-section"><div class="detail-label">Relationships ('+rels.length+')</div>'+
    rels.map(r=>'<div class="rel-item" onclick="focusNode('+r.other?.id+')">'+
      (r.direction==="out"?node.label+' <span class="rel-arrow">&rarr;</span> '+(r.other?.label||"?"):(r.other?.label||"?")+' <span class="rel-arrow">&rarr;</span> '+node.label)+
      '<span class="rel-desc">'+(r.edge._label||"")+'</span></div>').join("")+'</div>';
}
function showEdgeDetail(edgeId) {
  const edge=allEdges.get(edgeId); if(!edge)return;
  const f=allNodes.get(edge.from),t=allNodes.get(edge.to);
  document.getElementById("detail-header").textContent="Relationship";
  document.getElementById("detail-content").innerHTML=
    '<div class="detail-section"><div class="detail-label">Relationship</div><div class="detail-value" style="font-size:16px;"><span style="color:'+getColor(f?._type)+'">'+(f?.label||"?")+'</span> <span class="rel-arrow">&rarr;</span> <span style="color:'+getColor(t?._type)+'">'+(t?.label||"?")+'</span></div></div>'+
    '<div class="detail-section"><div class="detail-label">Description</div><div class="detail-value">'+(edge._label||"N/A")+'</div></div>'+
    '<div class="detail-section"><div class="detail-label">Weight</div><div class="detail-value">'+edge._weight+'</div></div>';
}
init();
</script>
</body>
</html>'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ─── main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG 索引基准测试（独立于正式数据）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python bench_index.py index --count 100 --embedding bge
  python bench_index.py index --count 500 --embedding v4
  python bench_index.py query
  python bench_index.py query --view [--top 200] [--port 8081]
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # index
    p_index = sub.add_parser("index", help="运行索引测试")
    p_index.add_argument("--count", type=int, default=100, help="文档数量 (默认: 100)")
    p_index.add_argument("--embedding", choices=["bge", "v4"], default="bge",
                         help="嵌入模型: bge=BGE-M3(本地), v4=text-embedding-v4(阿里云)")

    # query (含 --view 可视化)
    p_query = sub.add_parser("query", help="交互式查询 / 知识图谱可视化")
    p_query.add_argument("--view", action="store_true", help="启动知识图谱可视化（替代交互式查询）")
    p_query.add_argument("--top", type=int, default=0, help="可视化时只显示度数最高的 N 个实体 (0=全部)")
    p_query.add_argument("--port", type=int, default=8081, help="可视化 HTTP 端口 (默认: 8081)")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
