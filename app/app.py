"""
PestiGraph — 农药知识图谱智能问答系统
基于 Gradio + GraphRAG，部署到 Hugging Face Spaces
"""
import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import gradio as gr
import litellm
import pandas as pd

import graphrag.api as graphrag_api
from graphrag.config.load_config import load_config
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider
from graphrag.data_model.data_reader import DataReader

from query_router import route, MODE_LABELS
import graph_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# GraphRAG 内部日志
logging.getLogger("graphrag").setLevel(logging.INFO)
logging.getLogger("graphrag_llm").setLevel(logging.INFO)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("graphrag.query.llm.text_utils").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# ─── 实时日志缓冲 ─────────────────────────────────────────────
_query_log = deque(maxlen=200)
_llm_call_count = 0
_llm_total_tokens = 0
_map_call_count = 0
_map_total = 0
_drift_action_count = 0
_current_stage = "idle"     # idle|route|load|pack|map|reduce|local_search|basic_search|drift_hyde|drift_primer|drift_action|drift_reduce|done|error
_current_method = "global"  # global | local | drift | basic
_query_elapsed = 0.0
_query_start_time = 0.0


def _log(msg: str):
    """写入一条带时间戳的日志"""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _query_log.append(line)
    logger.info(msg)


def _get_log_html() -> str:
    """生成带可视化流水线的日志 HTML"""
    if _current_stage in ("idle", "done", "error"):
        elapsed = _query_elapsed
    else:
        elapsed = time.time() - _query_start_time if _query_start_time > 0 else 0.0
    return _build_pipeline_html(
        stage=_current_stage,
        method=_current_method,
        map_progress=_map_call_count,
        map_total=_map_total,
        drift_actions=_drift_action_count,
        llm_calls=_llm_call_count,
        tokens=_llm_total_tokens,
        elapsed=elapsed,
        log_lines=list(_query_log),
    )


def _build_pipeline_html(stage, method, map_progress, map_total,
                         drift_actions, llm_calls, tokens, elapsed, log_lines) -> str:
    """构建带实时流程图的日志 HTML"""
    import html as html_mod

    # 每种模式的 stage 顺序
    _STAGE_ORDER = {
        "global": ["route", "load", "pack", "map", "reduce", "done"],
        "local":  ["route", "load", "local_search", "done"],
        "drift":  ["route", "load", "drift_hyde", "drift_primer", "drift_action", "drift_reduce", "done"],
        "basic":  ["route", "load", "basic_search", "done"],
    }
    order = _STAGE_ORDER.get(method, _STAGE_ORDER["local"])

    def _cls(node_stage):
        if stage == "idle":
            return "nd-wait"
        if stage == "error":
            return "nd-err"
        if node_stage == stage:
            return "nd-active"
        try:
            ci = order.index(stage)
            ni = order.index(node_stage)
            return "nd-done" if ni < ci else "nd-wait"
        except ValueError:
            return "nd-wait"

    def _arrow_cls(from_stage):
        if stage == "idle":
            return "ar-wait"
        try:
            ci = order.index(stage)
            fi = order.index(from_stage)
            return "ar-done" if fi < ci else "ar-wait"
        except ValueError:
            return "ar-wait"

    def _nd(icon, text, node_stage, extra_cls="", sub=""):
        sub_html = f'<div class="nd-sub">{sub}</div>' if sub else ""
        return f'<div class="nd {extra_cls} {_cls(node_stage)}"><div class="nd-icon">{icon}</div><div class="nd-text">{text}</div>{sub_html}</div>'

    def _ar(from_stage):
        return f'<div class="ar {_arrow_cls(from_stage)}"><div class="ar-line"></div><div class="ar-head"></div></div>'

    # ── 根据查询模式生成不同流程图 ──
    if method == "global":
        map_pct = int(map_progress / map_total * 100) if map_total > 0 else 0
        map_info = f"{map_progress}/{map_total}" if map_total > 0 else ""
        map_bar = f'<div class="nd-bar"><div class="nd-fill" style="width:{map_pct}%"></div></div>' if stage == "map" and map_total > 0 else ""

        n_fan = min(map_total, 4) if map_total > 0 else 3
        fan_labels = [f"Map #{i+1}" for i in range(n_fan)]
        if map_total > n_fan:
            fan_labels[-1] = f"... #{map_total}"
        fan_items = ""
        for i, fl in enumerate(fan_labels):
            if stage in ("reduce", "done"):
                fc = "nd-done"
            elif stage == "map" and map_progress > i:
                fc = "nd-done"
            elif stage == "map" and map_progress == i:
                fc = "nd-active"
            elif stage == "map":
                fc = "nd-run"
            else:
                fc = "nd-wait"
            fan_items += f'<div class="fan-item {fc}">{fl}</div>'

        flow_html = f'''
<div class="flow">
  <div class="flow-row flow-main">
    {_nd("🔀", "智能路由", "route")}
    {_ar("route")}
    {_nd("📂", "加载数据", "load")}
    {_ar("load")}
    {_nd("📦", "报告打包", "pack")}
    {_ar("pack")}
    <div class="nd nd-wide {_cls("map")}">
      <div class="nd-icon">⚡</div>
      <div class="nd-text">Map 并发分析</div>
      <div class="nd-sub">{map_info}</div>
      {map_bar}
    </div>
    {_ar("map")}
    {_nd("📊", "Reduce 汇总", "reduce")}
    {_ar("reduce")}
    {_nd("✅", "完成", "done")}
  </div>
  <div class="fan-row">
    <div class="fan-spacer"></div>
    <div class="fan-bracket"></div>
    <div class="fan-items">{fan_items}</div>
  </div>
</div>'''

    elif method == "drift":
        drift_sub = f"×{drift_actions}" if drift_actions > 0 else ""
        flow_html = f'''
<div class="flow">
  <div class="flow-row flow-main">
    {_nd("🔀", "智能路由", "route")}
    {_ar("route")}
    {_nd("📂", "加载数据", "load")}
    {_ar("load")}
    {_nd("💡", "HyDE 扩展", "drift_hyde")}
    {_ar("drift_hyde")}
    {_nd("🎯", "社区定位", "drift_primer")}
    {_ar("drift_primer")}
    {_nd("🔬", "逐层检索", "drift_action", sub=drift_sub)}
    {_ar("drift_action")}
    {_nd("📊", "汇总生成", "drift_reduce")}
    {_ar("drift_reduce")}
    {_nd("✅", "完成", "done")}
  </div>
</div>'''

    elif method == "local":
        grp_cls = _cls("local_search")
        flow_html = f'''
<div class="flow">
  <div class="flow-row flow-main">
    {_nd("🔀", "智能路由", "route")}
    {_ar("route")}
    {_nd("📂", "加载数据", "load")}
    {_ar("load")}
    <div class="nd-group">
      <div class="nd {grp_cls}"><div class="nd-icon">🔍</div><div class="nd-text">向量检索</div></div>
      <div class="ar-mini {_arrow_cls("local_search")}">›</div>
      <div class="nd {grp_cls}"><div class="nd-icon">🧩</div><div class="nd-text">构建上下文</div></div>
      <div class="ar-mini {_arrow_cls("local_search")}">›</div>
      <div class="nd {grp_cls}"><div class="nd-icon">🤖</div><div class="nd-text">LLM 生成</div></div>
    </div>
    {_ar("local_search")}
    {_nd("✅", "完成", "done")}
  </div>
  <div class="grp-label">GraphRAG 内部处理</div>
</div>'''

    else:  # basic
        grp_cls = _cls("basic_search")
        flow_html = f'''
<div class="flow">
  <div class="flow-row flow-main">
    {_nd("🔀", "智能路由", "route")}
    {_ar("route")}
    {_nd("📂", "加载数据", "load")}
    {_ar("load")}
    <div class="nd-group">
      <div class="nd {grp_cls}"><div class="nd-icon">🔍</div><div class="nd-text">文本检索</div></div>
      <div class="ar-mini {_arrow_cls("basic_search")}">›</div>
      <div class="nd {grp_cls}"><div class="nd-icon">🤖</div><div class="nd-text">LLM 生成</div></div>
    </div>
    {_ar("basic_search")}
    {_nd("✅", "完成", "done")}
  </div>
  <div class="grp-label">GraphRAG 内部处理</div>
</div>'''

    # 统计栏
    if stage == "idle":
        stats_html = '<div class="st">等待查询...</div>'
    else:
        stats_html = (
            f'<div class="st">'
            f'LLM 调用: <b>{llm_calls}</b> 次 &nbsp;·&nbsp; '
            f'Tokens: <b>{tokens:,}</b> &nbsp;·&nbsp; '
            f'耗时: <b>{elapsed:.1f}</b>s'
            f'</div>'
        )

    log_html = "".join(
        f'<div class="ll">{html_mod.escape(line)}</div>'
        for line in log_lines
    )

    return f'''<div class="pr">
<style>
.pr{{font-family:"Microsoft YaHei",sans-serif;background:#0f0f1a !important;border-radius:10px;padding:14px;color:#c0c0d0 !important;}}
.flow{{padding:6px 0 2px;}}
.flow-row{{display:flex;align-items:center;justify-content:center;flex-wrap:nowrap;}}
/* 节点 */
.nd{{display:flex;flex-direction:column;align-items:center;padding:6px 8px;border:1.5px solid #3a3a55;border-radius:8px;background:#1a1b26 !important;min-width:56px;transition:all .3s;position:relative;}}
.nd-wide{{min-width:85px;}}
.nd-icon{{font-size:15px;line-height:1;}}
.nd-text{{font-size:10px;margin-top:2px;white-space:nowrap;color:#c0caf5 !important;}}
.nd-sub{{font-size:9px;color:#7aa2f7 !important;margin-top:1px;}}
.nd-bar{{width:65px;height:3px;background:#2a2a40 !important;border-radius:2px;margin-top:3px;overflow:hidden;}}
.nd-fill{{height:100%;background:#7aa2f7 !important;border-radius:2px;transition:width .3s;}}
/* 状态 */
.nd-wait{{opacity:.45;}}
.nd-active{{border-color:#7aa2f7 !important;background:#1e2540 !important;box-shadow:0 0 14px rgba(122,162,247,.35);animation:glow 2s infinite;}}
.nd-active .nd-text{{color:#7aa2f7 !important;font-weight:bold;}}
.nd-done{{border-color:#9ece6a !important;background:#1a2520 !important;}}
.nd-done .nd-text{{color:#9ece6a !important;}}
.nd-run{{border-color:#565f89;background:#1a1b26 !important;opacity:.7;}}
.nd-err{{border-color:#f7768e !important;background:#261a1e !important;}}
.nd-err .nd-text{{color:#f7768e !important;}}
@keyframes glow{{0%,100%{{box-shadow:0 0 8px rgba(122,162,247,.25);}}50%{{box-shadow:0 0 18px rgba(122,162,247,.55);}}}}
/* 箭头 */
.ar{{display:flex;align-items:center;width:20px;flex-shrink:0;}}
.ar-line{{flex:1;height:2px;background:#3a3a55 !important;}}
.ar-head{{width:0;height:0;border-top:4px solid transparent;border-bottom:4px solid transparent;border-left:6px solid #3a3a55;}}
.ar-done .ar-line{{background:#9ece6a !important;}}
.ar-done .ar-head{{border-left-color:#9ece6a !important;}}
/* 节点组（黑盒） */
.nd-group{{display:flex;align-items:center;border:1px dashed #3a3a5588;border-radius:10px;padding:3px 5px;background:#14152240;gap:0;}}
.ar-mini{{font-size:12px;color:#3a3a55 !important;margin:0 1px;font-weight:bold;}}
.ar-mini.ar-done{{color:#9ece6a !important;}}
.grp-label{{text-align:center;font-size:8px;color:#565f89 !important;margin-top:2px;}}
/* Map 扇出 */
.fan-row{{display:flex;align-items:flex-start;justify-content:center;padding:4px 0 0;}}
.fan-spacer{{width:220px;flex-shrink:0;}}
.fan-bracket{{width:2px;height:28px;border-left:2px dashed #3a3a55;margin:0 8px;}}
.fan-items{{display:flex;gap:4px;flex-wrap:wrap;align-items:flex-start;}}
.fan-item{{font-size:9px;padding:2px 7px;border-radius:4px;border:1px solid #3a3a55;background:#1a1b26 !important;color:#c0caf5 !important;white-space:nowrap;}}
.fan-item.nd-done{{border-color:#9ece6a !important;color:#9ece6a !important;background:#1a2520 !important;}}
.fan-item.nd-active{{border-color:#7aa2f7 !important;color:#7aa2f7 !important;background:#1e2540 !important;animation:glow 2s infinite;}}
.fan-item.nd-run{{border-color:#565f89;color:#7aa2f7 !important;opacity:.8;}}
/* 统计与日志 */
.st{{text-align:center;font-size:11px;color:#c0c0d0 !important;padding:6px 0 6px;border-top:1px solid #2a2a40;border-bottom:1px solid #2a2a40;margin:6px 0;}}
.st b{{color:#e0e8ff !important;}}
.la{{max-height:340px;overflow-y:auto;padding:2px 0;}}
.ll{{font-size:11px;line-height:1.55;padding:0 4px;font-family:"Cascadia Code","Consolas",monospace;color:#e0e0f0 !important;white-space:pre-wrap;word-break:break-all;}}
</style>
{flow_html}
{stats_html}
<div class="la" id="log-scroll">{log_html}</div>
<script>var e=document.getElementById("log-scroll");if(e)e.scrollTop=e.scrollHeight;</script>
</div>'''


# ─── MiMo 调用耗时监控 ─────────────────────────────────────────

def _detect_llm_phase(kwargs) -> str:
    """根据 system/user prompt 内容识别 LLM 调用阶段（含 Drift 子阶段）"""
    messages = kwargs.get("messages") or []
    sys_msg = ""
    user_msg = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system" and not sys_msg:
            sys_msg = content
        elif role == "user" and not user_msg:
            user_msg = content
    # 检测顺序：从具体到通用，避免误判
    if "rate how relevant" in sys_msg:
        return "评分筛选"
    if "Create a hypothetical answer" in user_msg:
        return "Drift HyDE"
    if "reason over a knowledge graph" in user_msg:
        return "Drift 定位"
    if "Data Reports" in sys_msg:
        return "Drift Reduce"
    if "multiple analysts" in sys_msg or "Analyst Reports" in sys_msg:
        return "Global Reduce"
    if "follow_up_queries" in sys_msg:
        return "Drift 检索"
    if "the tables" in sys_msg:
        return "Map 分析"
    return "LLM"


class _LLMTimingCallback(litellm.integrations.custom_logger.CustomLogger):
    """记录每次 LLM 调用的耗时和 token 用量"""

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        global _llm_call_count, _llm_total_tokens, _map_call_count, _current_stage, _drift_action_count
        elapsed = (end_time - start_time).total_seconds()
        usage = getattr(response_obj, "usage", None)
        _llm_call_count += 1
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        _llm_total_tokens += in_tok + out_tok
        phase = _detect_llm_phase(kwargs)
        progress = ""
        # 按当前模式分发 stage 更新
        if _current_method == "global":
            if phase == "Map 分析":
                _map_call_count += 1
                _current_stage = "map"
                if _map_total > 0:
                    progress = f" ({_map_call_count}/{_map_total})"
            elif phase == "Global Reduce":
                _current_stage = "reduce"
        elif _current_method == "drift":
            if phase == "Drift HyDE":
                _current_stage = "drift_hyde"
            elif phase == "Drift 定位":
                _current_stage = "drift_primer"
            elif phase == "Drift 检索":
                _drift_action_count += 1
                _current_stage = "drift_action"
                progress = f" (×{_drift_action_count})"
            elif phase == "Drift Reduce":
                _current_stage = "drift_reduce"
        _log(f"  ↳ [{phase}]{progress} #{_llm_call_count}: {elapsed:.1f}s | in={in_tok}/out={out_tok}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)

litellm.callbacks = [_LLMTimingCallback()]

# ─── 数据目录 ───────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"


# ─── GraphRAG 数据加载（启动时一次性加载） ──────────────────────

_config = None
_dataframes = {}


def _init_graphrag():
    """加载 GraphRAG 配置和索引数据"""
    global _config, _dataframes
    if _config is not None:
        return

    _config = load_config(root_dir=DATA_DIR)

    storage_obj = create_storage(_config.output_storage)
    table_provider = create_table_provider(_config.table_provider, storage=storage_obj)
    reader = DataReader(table_provider)

    tables = ["entities", "communities", "community_reports", "text_units", "relationships"]
    for name in tables:
        _dataframes[name] = asyncio.run(getattr(reader, name)())

    # covariates 是可选的
    try:
        has_cov = asyncio.run(table_provider.has("covariates"))
        if has_cov:
            _dataframes["covariates"] = asyncio.run(reader.covariates())
        else:
            _dataframes["covariates"] = None
    except Exception:
        _dataframes["covariates"] = None

    logger.info(
        "GraphRAG loaded: %d entities, %d relationships",
        len(_dataframes["entities"]),
        len(_dataframes["relationships"]),
    )


# ─── 查询函数 ───────────────────────────────────────────────────

async def _run_query(method: str, question: str,
                     community_level: int = 2,
                     response_type: str = "multiple paragraphs") -> str:
    """调用 GraphRAG API 执行查询"""
    _init_graphrag()

    common = {
        "config": _config,
        "entities": _dataframes["entities"],
        "communities": _dataframes["communities"],
        "community_reports": _dataframes["community_reports"],
        "community_level": community_level,
        "response_type": response_type,
        "query": question,
    }

    if method == "global":
        # 报告数少时全量扫描更可靠；报告多时启用动态筛选避免数千次调用
        report_count = len(_dataframes["community_reports"])
        use_dynamic = report_count > 200
        response, _ = await graphrag_api.global_search(
            dynamic_community_selection=use_dynamic,
            **common,
        )
    elif method == "local":
        response, _ = await graphrag_api.local_search(
            text_units=_dataframes["text_units"],
            relationships=_dataframes["relationships"],
            covariates=_dataframes["covariates"],
            **common,
        )
    elif method == "drift":
        response, _ = await graphrag_api.drift_search(
            config=_config,
            entities=_dataframes["entities"],
            communities=_dataframes["communities"],
            community_reports=_dataframes["community_reports"],
            text_units=_dataframes["text_units"],
            relationships=_dataframes["relationships"],
            community_level=common["community_level"] or 2,
            response_type=common["response_type"],
            query=question,
        )
    else:  # basic
        response, _ = await graphrag_api.basic_search(
            config=_config,
            text_units=_dataframes["text_units"],
            response_type=common["response_type"],
            query=question,
        )

    return str(response)


RESPONSE_TYPE_MAP = {
    "详细多段落": "multiple paragraphs",
    "简洁段落": "single paragraph",
    "要点列表": "list of 3-7 points",
    "一句话": "single sentence",
}


def chat(message: str, history: list,
         community_level: int = 2,
         response_type_label: str = "详细多段落",
         mode_override: str = "自动") -> str:
    """聊天回调：智能路由 + 查询"""
    global _llm_call_count, _llm_total_tokens, _map_call_count, _map_total
    global _current_stage, _current_method, _query_elapsed, _query_start_time, _drift_action_count

    if not message.strip():
        return "请输入您的问题。"

    # 重置计数器
    _llm_call_count = 0
    _llm_total_tokens = 0
    _map_call_count = 0
    _map_total = 0
    _drift_action_count = 0
    _query_elapsed = 0.0
    _query_start_time = time.time()
    _query_log.clear()

    # 阶段 1: 路由
    _current_stage = "route"
    _MODE_OVERRIDE_MAP = {"全局汇总": "global", "精确检索": "local", "关联分析": "drift", "基础检索": "basic"}
    if mode_override != "自动" and mode_override in _MODE_OVERRIDE_MAP:
        method = _MODE_OVERRIDE_MAP[mode_override]
        matched_kw = ""
        _current_method = method
        mode_label = MODE_LABELS.get(method, method)
        _log(f"📋 收到问题: {message}")
        _log(f"🔀 手动指定模式 → {mode_label} ({method})")
    else:
        method, matched_kw = route(message)
        _current_method = method
        mode_label = MODE_LABELS.get(method, method)
        _log(f"📋 收到问题: {message}")
        if matched_kw:
            _log(f"🔀 智能路由 → {mode_label} ({method})  匹配关键词: '{matched_kw}'")
        else:
            _log(f"🔀 智能路由 → {mode_label} ({method})  未匹配全局/关联词，走精确检索")
    response_type = RESPONSE_TYPE_MAP.get(response_type_label, "multiple paragraphs")
    _log(f"⚙️ 参数: 社区层级={community_level}, 回答格式={response_type_label}")

    try:
        # 阶段 2: 初始化
        _current_stage = "load"
        _log("📂 加载 GraphRAG 数据...")
        _init_graphrag()
        ent_n = len(_dataframes["entities"])
        rel_n = len(_dataframes["relationships"])
        comm_n = len(_dataframes["communities"])
        report_n = len(_dataframes["community_reports"])
        tu_n = len(_dataframes["text_units"])
        _log(f"✅ 数据就绪: 实体={ent_n} | 关系={rel_n} | 社区={comm_n} | 报告={report_n} | 文本片段={tu_n}")

        # 阶段 3: 查询
        if method == "global":
            _current_stage = "pack"
        elif method == "local":
            _current_stage = "local_search"
        elif method == "drift":
            _current_stage = "drift_hyde"
        else:
            _current_stage = "basic_search"
        _log(f"🔍 开始 {mode_label} 查询...")
        if method == "global":
            # 统计当前层级的社区报告数
            reports_df = _dataframes["community_reports"]
            communities_df = _dataframes["communities"]
            try:
                level_comm = communities_df[communities_df["level"] <= community_level]
                level_ids = set(level_comm["community"].unique())
                level_reports = reports_df[reports_df["community"].isin(level_ids)]
                _map_total = len(level_reports)
                _log(f"  → 社区层级 ≤{community_level}: {len(level_ids)} 个社区, {_map_total} 份报告")
            except Exception:
                _map_total = report_n
                _log(f"  → 共 {report_n} 份社区报告")
            _log(f"  → Map 阶段: 逐份分析社区报告 (预计 {_map_total} 次 LLM 调用)...")
            _log(f"  → Reduce 阶段: 将在 Map 完成后汇总所有结果")
        elif method == "local":
            _log("  → 向量检索相关实体 + 文本片段...")
            _log(f"  → 将基于检索结果调用 LLM 生成回答")
        elif method == "drift":
            _log("  → Drift 搜索: 跨社区层级关联分析...")
            _log(f"  → 先在高层社区定位，再逐层细化检索")
        else:
            _log("  → Basic 搜索: 文本片段向量检索...")
            _log(f"  → 将基于最相关文本片段调用 LLM 生成回答")

        start = time.time()
        response = asyncio.run(_run_query(method, message, community_level, response_type))
        elapsed = time.time() - start
        _query_elapsed = elapsed

        if method == "global":
            _current_stage = "reduce"
            _log(f"  → Reduce 阶段: 汇总 {_map_call_count} 份 Map 结果完成")

        # 阶段 4: 完成
        _current_stage = "done"
        avg_time = elapsed / _llm_call_count if _llm_call_count > 0 else 0
        _log(f"✅ 查询完成: 总耗时 {elapsed:.1f}s | LLM调用 {_llm_call_count}次 | 平均 {avg_time:.1f}s/次 | tokens: {_llm_total_tokens}")

        return f"{response}\n\n---\n*查询模式: {mode_label} | 社区层级: {community_level} | 回答格式: {response_type_label} | 耗时: {elapsed:.1f}s | LLM调用: {_llm_call_count}次*"
    except Exception as e:
        _current_stage = "error"
        _log(f"❌ 查询出错: {str(e)}")
        logger.exception("Query failed")
        return f"查询出错: {str(e)}"


# ─── 知识图谱 Tab ───────────────────────────────────────────────

def _build_architecture_html() -> str:
    """构建查询原理说明页面，含 Mermaid 流程图"""
    import base64

    # 注入当前数据统计
    try:
        _init_graphrag()
        ent_n = len(_dataframes["entities"])
        rel_n = len(_dataframes["relationships"])
        comm_df = _dataframes["communities"]
        rpt_df = _dataframes["community_reports"]
        comm_n = len(comm_df)
        report_n = len(rpt_df)
        tu_n = len(_dataframes["text_units"])

        # 每层社区数 / 有报告数
        rpt_comms = set(rpt_df["community"].unique())
        level_stats = {}
        for lv in sorted(comm_df["level"].unique()):
            lv_comms = set(comm_df[comm_df["level"] == lv]["community"].unique())
            level_stats[int(lv)] = (len(lv_comms), len(lv_comms & rpt_comms))
    except Exception:
        ent_n = rel_n = comm_n = report_n = tu_n = "?"
        level_stats = {}

    page = _ARCHITECTURE_PAGE.replace("__ENT_N__", str(ent_n))
    page = page.replace("__REL_N__", str(rel_n))
    page = page.replace("__COMM_N__", str(comm_n))
    page = page.replace("__REPORT_N__", str(report_n))
    page = page.replace("__TU_N__", str(tu_n))

    # 每层统计
    for lv in range(3):
        total, rpt = level_stats.get(lv, ("?", "?"))
        no_rpt = total - rpt if isinstance(total, int) else "?"
        page = page.replace(f"__LV{lv}_TOTAL__", str(total))
        page = page.replace(f"__LV{lv}_RPT__", str(rpt))
        page = page.replace(f"__LV{lv}_NORPT__", str(no_rpt))

    b64 = base64.b64encode(page.encode("utf-8")).decode("ascii")
    return f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:800px;border:none;border-radius:8px;"></iframe>'


_ARCHITECTURE_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#0f0f1a;color:#d8d8e8;line-height:1.7;padding:30px 40px;overflow-y:auto;}
h1{font-size:22px;color:#fff;margin-bottom:8px;}
h2{font-size:17px;color:#7aa2f7;margin:32px 0 12px;padding-left:10px;border-left:3px solid #7aa2f7;}
h3{font-size:15px;color:#9ece6a;margin:20px 0 8px;}
p,li{font-size:14px;color:#c0c0d0;}
ul{padding-left:20px;margin:6px 0 12px;}
li{margin:3px 0;}
code{background:#1a1b26;padding:1px 6px;border-radius:3px;font-size:13px;color:#e0af68;}
.subtitle{color:#888;font-size:13px;margin-bottom:24px;}
.stats{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0;}
.stat{background:#1a1b26;border:1px solid #2a2a40;border-radius:8px;padding:10px 16px;text-align:center;min-width:100px;}
.stat .num{font-size:22px;color:#7aa2f7;font-weight:bold;}
.stat .lbl{font-size:11px;color:#888;margin-top:2px;}
.card{background:#1a1b26;border:1px solid #2a2a40;border-radius:10px;padding:20px 24px;margin:14px 0;}
.mermaid{margin:16px 0;display:flex;justify-content:center;}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px;}
th{background:#1e1e35;color:#9aa5ce;padding:8px 12px;text-align:left;border:1px solid #2a2a40;}
td{padding:8px 12px;border:1px solid #2a2a40;color:#c0c0d0;}
tr:nth-child(even){background:#151525;}
.tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;color:#fff;margin-right:6px;}
.g{background:#3d59a1;} .l{background:#9ece6a;color:#1a1b26;} .d{background:#e0af68;color:#1a1b26;} .b{background:#565f89;}
.arrow{color:#7aa2f7;font-size:16px;}
</style>
</head>
<body>
<h1>PestiGraph 查询原理</h1>
<p class="subtitle">基于 Microsoft GraphRAG 的知识图谱增强检索生成系统</p>

<div class="stats">
  <div class="stat"><div class="num">__ENT_N__</div><div class="lbl">实体</div></div>
  <div class="stat"><div class="num">__REL_N__</div><div class="lbl">关系</div></div>
  <div class="stat"><div class="num">__COMM_N__</div><div class="lbl">社区</div></div>
  <div class="stat"><div class="num">__REPORT_N__</div><div class="lbl">社区报告</div></div>
  <div class="stat"><div class="num">__TU_N__</div><div class="lbl">文本片段</div></div>
</div>

<!-- ====== 1. 整体架构 ====== -->
<h2>1. 整体架构</h2>
<div class="card">
<p>系统分为<strong>离线索引</strong>和<strong>在线查询</strong>两个阶段：</p>
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart LR
    subgraph 离线索引
    A[200 篇农药标签] -->|分块| B[文本片段]
    B -->|LLM 提取| C[实体 + 关系]
    C -->|Leiden 聚类| D[社区层级树]
    D -->|LLM 摘要| E[社区报告]
    C -->|向量化| F[Embedding 索引]
    end
    subgraph 在线查询
    Q[用户问题] --> R{智能路由}
    R -->|汇总类| G[Global Search]
    R -->|具体查询| L[Local Search]
    R -->|关联比较| DR[Drift Search]
    G --> ANS[最终回答]
    L --> ANS
    DR --> ANS
    end
    E -.-> G
    F -.-> L
    E -.-> DR
</pre>
</div>

<!-- ====== 2. 智能路由 ====== -->
<h2>2. 智能路由</h2>
<div class="card">
<p>零成本、零延迟的关键词规则，根据问题自动选择最合适的查询模式：</p>
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart TD
    Q[用户问题] --> C1{含 '有哪些/总结/排名/统计/所有...'?}
    C1 -->|是| G[<strong>Global Search</strong><br/>全局汇总]
    C1 -->|否| C2{含 '关系/搭配/比较/替代/混用...'?}
    C2 -->|是| D[<strong>Drift Search</strong><br/>关联分析]
    C2 -->|否| L[<strong>Local Search</strong><br/>精确检索]
    style G fill:#3d59a1,stroke:#545c7e,color:#c0caf5
    style D fill:#e0af68,stroke:#545c7e,color:#1a1b26
    style L fill:#9ece6a,stroke:#545c7e,color:#1a1b26
</pre>
<table>
<tr><th>模式</th><th>触发关键词</th><th>适用场景</th><th>LLM 调用次数</th></tr>
<tr><td><span class="tag g">Global</span></td><td>有哪些、总结、排名、统计、所有、列举...</td><td>需要遍历全局信息的汇总类问题</td><td>多次（与社区报告数相关）</td></tr>
<tr><td><span class="tag l">Local</span></td><td>默认（无特殊关键词）</td><td>查具体产品、成分、登记证号</td><td>1~2 次</td></tr>
<tr><td><span class="tag d">Drift</span></td><td>关系、搭配、比较、替代、混用...</td><td>跨实体关联和对比分析</td><td>数次（跨层检索）</td></tr>
</table>
</div>

<!-- ====== 3. Global Search ====== -->
<h2>3. Global Search — Map-Reduce 全局查询</h2>
<div class="card">
<p>将问题广播到所有社区报告，再汇总结果。适合"有哪些""总结""统计"类问题。</p>
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart TD
    Q["用户问题：含有吡虫啉的农药有哪些？"] --> PACK[将社区报告按 token 上限打包]
    PACK --> M1["Map #1<br/>报告 1~8 → LLM"]
    PACK --> M2["Map #2<br/>报告 9~15 → LLM"]
    PACK --> M3["Map #3<br/>报告 16~22 → LLM"]
    PACK --> MN["...<br/>Map #N"]
    M1 --> |"相关片段 + 评分"| R[Reduce 汇总]
    M2 --> |"相关片段 + 评分"| R
    M3 --> |"相关片段 + 评分"| R
    MN --> |"相关片段 + 评分"| R
    R --> ANS[最终回答]
    style Q fill:#3d59a1,stroke:#545c7e,color:#c0caf5
    style R fill:#e0af68,stroke:#545c7e,color:#1a1b26
    style ANS fill:#9ece6a,stroke:#545c7e,color:#1a1b26
</pre>
<h3>执行细节</h3>
<ul>
<li><strong>Map 阶段</strong>：多份报告打包进一次 LLM 调用（按 token 上限），所有 Map 并发执行</li>
<li><strong>Reduce 阶段</strong>：将所有 Map 结果合并，由 LLM 生成最终结构化回答</li>
<li>报告数少（≤200）时全量扫描；报告数多时自动启用 Dynamic Community Selection 预筛选</li>
</ul>
</div>

<!-- ====== 4. Dynamic Community Selection ====== -->
<h2>4. Dynamic Community Selection — 动态社区筛选</h2>
<div class="card">
<p>当社区报告数量很大时，全量 Map 不可行。此机制从顶层社区自上而下逐层筛选，只保留相关社区进入 Map。</p>
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart TD
    START["开始：取 Level 0 根社区"] --> RATE["LLM 打分（0~5）"]
    RATE --> CHK{得分 >= 阈值?}
    CHK -->|是| MARK["✅ 标记相关"]
    CHK -->|否| DROP["❌ 丢弃（子树全部跳过）"]
    MARK --> CHILD["展开子社区到队列"]
    CHILD --> NEXT{还有下一层?}
    DROP --> NEXT
    NEXT -->|是| RATE
    NEXT -->|否| COLLECT["收集所有相关社区报告"]
    COLLECT --> MAP["送入 Map-Reduce"]

    style START fill:#3d59a1,stroke:#545c7e,color:#c0caf5
    style MARK fill:#9ece6a,stroke:#545c7e,color:#1a1b26
    style DROP fill:#f7768e,stroke:#545c7e,color:#1a1b26
    style MAP fill:#e0af68,stroke:#545c7e,color:#1a1b26
</pre>
<h3>打分 Prompt</h3>
<p>每个社区报告发给 LLM：</p>
<table>
<tr><th>输入</th><th>内容</th></tr>
<tr><td>信息</td><td>社区报告的完整内容（<code>use_summary=false</code>时）</td></tr>
<tr><td>问题</td><td>用户的原始问题</td></tr>
<tr><td>要求</td><td>返回 JSON <code>{"reason": "...", "rating": 0~5}</code></td></tr>
</table>
<h3>兜底机制</h3>
<p>如果某一层所有社区全部被判为不相关（rating &lt; 阈值），不会返回空结果，而是<strong>跳到下一层全量重评</strong>，防止粗粒度误判。</p>
<h3>配置参数</h3>
<table>
<tr><th>参数</th><th>当前值</th><th>说明</th></tr>
<tr><td><code>threshold</code></td><td>1</td><td>得分 ≥1 即保留，仅 0 分丢弃</td></tr>
<tr><td><code>use_summary</code></td><td>false</td><td>用完整报告打分（摘要太短易误判）</td></tr>
<tr><td><code>keep_parent</code></td><td>false</td><td>子社区命中时替换父社区，避免重复</td></tr>
<tr><td><code>num_repeats</code></td><td>1</td><td>每社区打分次数，多次取众数更准</td></tr>
<tr><td><code>max_level</code></td><td>2</td><td>最深探索层级</td></tr>
</table>

<h3>真实案例：吡虫啉查询的失败链路</h3>
<p>问题"含有吡虫啉的农药有哪些？"时，详细讲吡虫啉的 5 个社区（139~143）位于 Level 2。它们的祖先链路：</p>
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart TD
    Q["问题: 含有吡虫啉的农药有哪些?"]
    C4["社区 4 (Level 0)<br/>size=214 | ❌ 无报告"]
    C55["社区 55 (Level 1)<br/>size=52 | ❌ 无报告"]
    C139["社区 139 (Level 2)<br/>size=44 | ✅ 吡虫啉核心"]
    C140["社区 140~143<br/>共 4 个吡虫啉相关社区"]
    SKIP["❌ 整棵子树被跳过<br/>无法触达吡虫啉信息"]

    Q --> C4
    C4 -->|"Level 0 无报告<br/>无法评分"| SKIP
    C4 -.->|"如果能展开..."| C55
    C55 -.-> C139
    C55 -.-> C140

    style Q fill:#3d59a1,stroke:#545c7e,color:#c0caf5
    style C4 fill:#f7768e,stroke:#545c7e,color:#1a1b26
    style C55 fill:#f7768e,stroke:#545c7e,color:#1a1b26
    style C139 fill:#9ece6a,stroke:#545c7e,color:#1a1b26
    style C140 fill:#9ece6a,stroke:#545c7e,color:#1a1b26
    style SKIP fill:#565f89,stroke:#545c7e,color:#c0caf5
</pre>
<p><strong>问题根源</strong>：社区 4（根）→ 55（中间）→ 139~143（叶），<strong>三层全无报告</strong>。动态筛选从 Level 0 开始打分，但社区 4 没有报告可以评分，这条分支在起点就被跳过，导致 5 个吡虫啉核心社区永远不会被访问到。</p>
<p>此外，Level 0 只有 __LV0_RPT__ 个根社区有报告，砍掉任何一个都可能损失 &gt;10% 的信息——对 200 篇小数据集来说代价太大。因此当前配置<strong>禁用</strong>了动态筛选，全量扫描 __REPORT_N__ 份报告。</p>
</div>

<!-- ====== 5. Local Search ====== -->
<h2>5. Local Search — 精确检索</h2>
<div class="card">
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart LR
    Q[用户问题] --> EMB[问题向量化]
    EMB --> VS["向量检索<br/>匹配最近实体"]
    VS --> CTX["构建上下文<br/>实体 + 关系 + 文本片段 + 社区报告"]
    CTX --> LLM["LLM 生成回答"]
    LLM --> ANS[最终回答]
    style Q fill:#3d59a1,stroke:#545c7e,color:#c0caf5
    style ANS fill:#9ece6a,stroke:#545c7e,color:#1a1b26
</pre>
<ul>
<li>将问题向量化，在 LanceDB 中检索最相似的实体</li>
<li>围绕命中实体收集关联的关系、文本片段、所属社区报告</li>
<li>拼装上下文后一次 LLM 调用生成回答（1~2 次调用）</li>
<li>适合查具体产品、成分、企业等明确实体的问题</li>
</ul>
</div>

<!-- ====== 6. Drift Search ====== -->
<h2>6. Drift Search — 跨社区关联分析</h2>
<div class="card">
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart TD
    Q[用户问题] --> TOP["高层社区定位<br/>（粗粒度匹配）"]
    TOP --> MID["中层社区细化<br/>（缩小范围）"]
    MID --> BOT["底层社区精查<br/>（收集证据）"]
    BOT --> REDUCE["汇总多层结果"]
    REDUCE --> ANS[最终回答]
    style Q fill:#3d59a1,stroke:#545c7e,color:#c0caf5
    style ANS fill:#9ece6a,stroke:#545c7e,color:#1a1b26
</pre>
<ul>
<li>从高层社区开始定位，逐层向下细化，跨越多个社区收集信息</li>
<li>适合"A 和 B 的关系""替代方案""搭配使用"等需要关联多个实体的问题</li>
<li>LLM 调用次数取决于层级深度和匹配的社区数</li>
</ul>
</div>

<!-- ====== 7. 社区层级 ====== -->
<h2>7. Leiden 社区层级结构</h2>
<div class="card">
<h3>什么是社区？</h3>
<p>知识图谱是一张由实体（节点）和关系（边）构成的大网。Leiden 算法的作用是把<strong>连接紧密的节点分到同一组</strong>，这一组就叫"社区"。</p>
<p>直觉理解：社区就像<strong>产业聚落</strong>——经常一起出现的产品、成分、企业、作物自然聚成一团，形成一个主题聚类。例如：</p>
<ul>
<li><strong>社区 5</strong> (size=77)：烟嘧磺隆 + 玉米除草剂 + 相关杂草 + 登记企业 + 登记证号</li>
<li><strong>社区 0</strong> (size=154)：杀虫剂集群——小菜蛾防治、甲氨基阿维菌素、高氯·甲维盐等</li>
<li><strong>社区 10</strong>：杀菌剂集群——苯醚甲环唑、肟菌·戊唑醇、氟嘧菌酯等</li>
</ul>
<p>每个社区由 LLM 生成一份<strong>社区报告</strong>（摘要），Global Search 就是逐份分析这些报告来回答问题。</p>

<h3>层级结构</h3>
<p>Leiden 算法产出的不是一层，而是一棵<strong>层级树</strong>：大社区套小社区。</p>
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart TD
    L0["Level 0（根层）<br/>__LV0_TOTAL__ 个大社区，覆盖面广"]
    L1["Level 1（中层）<br/>__LV1_TOTAL__ 个中社区，主题更聚焦"]
    L2["Level 2（叶层）<br/>__LV2_TOTAL__ 个小社区，颗粒度最细"]
    L0 --> L1
    L1 --> L2
    style L0 fill:#3d59a1,stroke:#545c7e,color:#c0caf5
    style L1 fill:#565f89,stroke:#545c7e,color:#c0caf5
    style L2 fill:#24283b,stroke:#545c7e,color:#c0caf5
</pre>

<h3>为什么 __COMM_N__ 个社区只有 __REPORT_N__ 份报告？</h3>
<p>大社区内容超过 <code>max_input_length</code>（8000 tokens）时，其报告<strong>由子社区报告替代</strong>，不会重复生成：</p>
<table>
<tr><th>层级</th><th>社区数</th><th>有报告</th><th>无报告</th><th>无报告原因</th></tr>
<tr><td>Level 0</td><td>__LV0_TOTAL__</td><td>__LV0_RPT__</td><td>__LV0_NORPT__</td><td>size 58~214，超 token 限制被子社区替代</td></tr>
<tr><td>Level 1</td><td>__LV1_TOTAL__</td><td>__LV1_RPT__</td><td>__LV1_NORPT__</td><td>size 24~52，仍超限</td></tr>
<tr><td>Level 2</td><td>__LV2_TOTAL__</td><td>__LV2_RPT__</td><td>__LV2_NORPT__</td><td>少数仍超限</td></tr>
<tr style="font-weight:bold;"><td>合计</td><td>__COMM_N__</td><td>__REPORT_N__</td><td></td><td></td></tr>
</table>

<h3>社区树展开示例</h3>
<p>以<strong>社区 0</strong>（size=154，杀虫剂大类）为例，它太大无法生成单一报告，被拆成 10 个子社区：</p>
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart TD
    C0["社区 0 (Level 0)<br/>size=154 | ❌ 无报告<br/>杀虫剂大类"]
    C20["社区 20 (Level 1)<br/>size=20 | ✅ 有报告<br/>小菜蛾防治"]
    C21["社区 21<br/>size=21 | ✅ 有报告<br/>甲氨基阿维菌素"]
    C22["社区 22<br/>size=19 | ✅ 有报告<br/>高氯·甲维盐"]
    C23["社区 23<br/>size=21 | ❌ 无报告"]
    C24["社区 24<br/>size=9 | ✅ 有报告<br/>阿维·高氯氟"]
    MORE["... 共 10 个子社区"]
    C0 --> C20
    C0 --> C21
    C0 --> C22
    C0 --> C23
    C0 --> C24
    C0 --> MORE
    style C0 fill:#f7768e,stroke:#545c7e,color:#1a1b26
    style C20 fill:#9ece6a,stroke:#545c7e,color:#1a1b26
    style C21 fill:#9ece6a,stroke:#545c7e,color:#1a1b26
    style C22 fill:#9ece6a,stroke:#545c7e,color:#1a1b26
    style C23 fill:#f7768e,stroke:#545c7e,color:#1a1b26
    style C24 fill:#9ece6a,stroke:#545c7e,color:#1a1b26
    style MORE fill:#565f89,stroke:#545c7e,color:#c0caf5
</pre>
<p>红色 = 无报告（太大，由子社区替代），绿色 = 有独立报告。</p>

<h3>配置参数</h3>
<table>
<tr><th>参数</th><th>值</th><th>含义</th></tr>
<tr><td><code>community_level</code></td><td>0~2</td><td>查询时使用的最大层级，越大覆盖越细但越慢</td></tr>
<tr><td><code>max_cluster_size</code></td><td>30</td><td>单个社区最大节点数，越小社区越多、层级越深</td></tr>
<tr><td><code>max_input_length</code></td><td>8000</td><td>单份报告的 token 上限，超过则由子社区替代</td></tr>
</table>
</div>

<!-- ====== 8. 数据规模与策略 ====== -->
<h2>8. 数据规模与策略选择</h2>
<div class="card">
<p>全量扫描和动态筛选各有适用场景，关键变量是<strong>社区报告数量</strong>：</p>
<table>
<tr><th>数据规模</th><th>社区报告数</th><th>全量 Map-Reduce</th><th>Dynamic Selection</th></tr>
<tr><td>小（~200 篇文档）</td><td>~81</td><td>✅ 11 次并发 LLM，几十秒完成</td><td>❌ 根社区太少，过滤太激进，丢信息</td></tr>
<tr><td>中（~2000 篇文档）</td><td>~500</td><td>⚠️ ~60 次 LLM，1~2 分钟</td><td>✅ 筛到 50~100，效果好</td></tr>
<tr><td>大（~5 万篇文档）</td><td>数千</td><td>❌ 数百次 LLM，不可行</td><td>✅ 筛到 100~200，必须启用</td></tr>
</table>
<h3>当前系统的自动策略</h3>
<pre class="mermaid">
%%{init:{'theme':'dark','themeVariables':{'primaryColor':'#3d59a1','primaryTextColor':'#c0caf5','primaryBorderColor':'#545c7e','lineColor':'#545c7e','secondaryColor':'#1a1b26','tertiaryColor':'#24283b','background':'#1a1b26','mainBkg':'#1a1b26','nodeBorder':'#545c7e','clusterBkg':'#1a1b26','clusterBorder':'#3d59a1','titleColor':'#c0caf5','edgeLabelBackground':'#1a1b26','nodeTextColor':'#c0caf5'}}}%%
flowchart LR
    RPT["社区报告数 = __REPORT_N__"] --> CHK{"报告数 > 200?"}
    CHK -->|"否（当前）"| FULL["全量 Map-Reduce<br/>扫描所有报告"]
    CHK -->|"是（扩容后）"| DYN["Dynamic Selection<br/>先筛选再 Map"]
    style FULL fill:#9ece6a,stroke:#545c7e,color:#1a1b26
    style DYN fill:#3d59a1,stroke:#545c7e,color:#c0caf5
</pre>
<p>代码逻辑：<code>dynamic_community_selection = report_count > 200</code>，无需手动切换。</p>
</div>

<script>mermaid.initialize({startOnLoad:true,theme:'dark',securityLevel:'loose'});</script>
</body>
</html>'''


def _build_graph_viewer_html() -> str:
    """构建知识图谱浏览器，用 iframe srcdoc 绕过 Gradio 的 script 过滤"""
    graph_data = graph_api.get_full_graph()
    graph_json = json.dumps(graph_data, ensure_ascii=False)

    # 构建完整 HTML 页面，通过 base64 data URI 嵌入 iframe
    import base64
    page = _GRAPH_VIEWER_PAGE.replace("__GRAPH_DATA__", graph_json)
    b64 = base64.b64encode(page.encode("utf-8")).decode("ascii")
    return f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:750px;border:none;border-radius:8px;"></iframe>'


_GRAPH_VIEWER_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:"Microsoft YaHei",sans-serif;background:#0f0f1a;color:#ddd;display:flex;height:100vh;overflow:hidden;}
#left{width:240px;min-width:240px;background:#161625;border-right:1px solid #2a2a40;display:flex;flex-direction:column;overflow:hidden;}
#left h2{padding:12px;font-size:15px;color:#fff;border-bottom:1px solid #2a2a40;}
#search{margin:8px 10px;padding:7px 10px;background:#1e1e35;border:1px solid #3a3a55;border-radius:6px;color:#fff;font-size:13px;outline:none;}
#search:focus{border-color:#5a7aff;}
#filters{padding:6px 10px;border-bottom:1px solid #2a2a40;max-height:180px;overflow-y:auto;}
.fi{display:flex;align-items:center;padding:2px 0;cursor:pointer;font-size:12px;}
.fi input{margin-right:6px;}
.fd{width:9px;height:9px;border-radius:50%;margin-right:5px;display:inline-block;}
.fc{margin-left:auto;color:#666;font-size:11px;}
#list{flex:1;overflow-y:auto;padding:6px 0;}
.gt{padding:4px 10px;font-size:11px;color:#888;position:sticky;top:0;background:#161625;}
.gi{padding:4px 10px 4px 20px;font-size:12px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.gi:hover{background:#1e1e35;}
#mid{flex:1;position:relative;}
#graph{width:100%;height:100%;}
#stats{position:absolute;top:10px;left:10px;font-size:11px;color:#666;background:rgba(0,0,0,0.5);padding:4px 8px;border-radius:4px;}
#right{width:280px;min-width:280px;background:#161625;border-left:1px solid #2a2a40;display:flex;flex-direction:column;overflow:hidden;}
#dhdr{padding:12px;font-size:15px;color:#fff;border-bottom:1px solid #2a2a40;}
#detail{flex:1;overflow-y:auto;padding:10px 14px;}
.ds{margin-bottom:12px;}
.dl{font-size:11px;color:#888;margin-bottom:4px;}
.dv{font-size:13px;line-height:1.5;}
.tb{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;color:#fff;}
.ri{padding:6px;margin:3px 0;background:#1e1e35;border-radius:5px;font-size:12px;cursor:pointer;line-height:1.4;}
.ri:hover{background:#252545;}
.rd{color:#999;font-size:11px;display:block;margin-top:2px;}
</style>
</head>
<body>
<div id="left">
  <h2>实体浏览</h2>
  <input id="search" type="text" placeholder="搜索实体...">
  <div id="filters"></div>
  <div id="list"></div>
</div>
<div id="mid">
  <div id="graph"></div>
  <div id="stats"></div>
</div>
<div id="right">
  <div id="dhdr">详情</div>
  <div id="detail"><p style="color:#666;font-size:13px;">点击节点或边查看详情</p></div>
</div>
<script>
var graphData = __GRAPH_DATA__;
var TC={"PRODUCT":"#e74c3c","ORGANIZATION":"#3498db","CHEMICAL":"#2ecc71","FORMULATION":"#f39c12","CROP":"#27ae60","PEST":"#e67e22","ID":"#9b59b6","DATE":"#8e44ad","CONCEPT":"#1abc9c"};
function gc(t){return TC[t]||TC[t&&t.toUpperCase()]||"#7f8c8d";}

var dMax=Math.max.apply(null,graphData.nodes.map(function(n){return n.degree;}))||1;
var wMax=Math.max.apply(null,graphData.edges.map(function(e){return e.weight;}))||1;

var nodesArr=graphData.nodes.map(function(n){
  return{id:n.id,label:n.label,_type:n.type,_desc:n.description,_freq:n.frequency,_deg:n.degree,
    size:8+Math.sqrt(n.degree/dMax)*28,
    color:{background:gc(n.type),border:gc(n.type),highlight:{background:"#fff",border:gc(n.type)}},
    font:{color:"#eee",size:11},borderWidth:1.5};
});
var edgesArr=graphData.edges.map(function(e,i){
  return{id:i,from:e.from,to:e.to,_sl:e.source_label,_tl:e.target_label,_lb:e.label,_w:e.weight,
    width:0.5+(e.weight/wMax)*3,arrows:"to",
    color:{color:"rgba(150,150,180,0.4)",highlight:"#5a7aff"},smooth:{type:"continuous"}};
});

var allNodes=new vis.DataSet(nodesArr);
var allEdges=new vis.DataSet(edgesArr);
var net=new vis.Network(document.getElementById("graph"),
  {nodes:allNodes,edges:allEdges},
  {physics:{solver:"barnesHut",barnesHut:{gravitationalConstant:-3000,centralGravity:0.2,springLength:120,springConstant:0.02},stabilization:{iterations:150,updateInterval:25}},
   interaction:{hover:true,tooltipDelay:200,hideEdgesOnDrag:true},
   layout:{improvedLayout:false}});
net.once("stabilizationIterationsDone",function(){net.setOptions({physics:{enabled:false}});});
document.getElementById("stats").textContent=graphData.nodes.length+" 实体 | "+graphData.edges.length+" 关系";

// 类型筛选
var counts={};
graphData.nodes.forEach(function(n){counts[n.type]=(counts[n.type]||0)+1;});
var fDiv=document.getElementById("filters");
var hidden={};
Object.keys(counts).sort(function(a,b){return counts[b]-counts[a];}).forEach(function(t){
  var d=document.createElement("label");d.className="fi";
  d.innerHTML='<input type="checkbox" checked data-t="'+t+'"><span class="fd" style="background:'+gc(t)+'"></span>'+t+'<span class="fc">'+counts[t]+'</span>';
  d.querySelector("input").addEventListener("change",function(){
    if(this.checked)delete hidden[this.dataset.t];else hidden[this.dataset.t]=1;
    allNodes.forEach(function(nd){allNodes.update({id:nd.id,hidden:!!hidden[nd._type]});});
    allEdges.forEach(function(ed){var f=allNodes.get(ed.from),t2=allNodes.get(ed.to);allEdges.update({id:ed.id,hidden:!f||!t2||f.hidden||t2.hidden});});
  });
  fDiv.appendChild(d);
});

// 实体列表
function buildList(filter){
  var c=document.getElementById("list");c.innerHTML="";
  var g={};
  graphData.nodes.forEach(function(n){
    if(filter&&n.label.indexOf(filter)<0)return;
    if(!g[n.type])g[n.type]=[];g[n.type].push(n);
  });
  Object.keys(g).sort().forEach(function(t){
    var h=document.createElement("div");h.className="gt";
    h.textContent=t+" ("+g[t].length+")";c.appendChild(h);
    g[t].sort(function(a,b){return b.degree-a.degree;}).forEach(function(n){
      var it=document.createElement("div");it.className="gi";
      it.textContent=n.label;
      it.onclick=function(){focusNode(n.id);};
      c.appendChild(it);
    });
  });
}
buildList();

var stimer;
document.getElementById("search").addEventListener("input",function(e){
  clearTimeout(stimer);var q=e.target.value.trim();
  stimer=setTimeout(function(){
    buildList(q||undefined);
    if(q){var m=graphData.nodes.find(function(n){return n.label.indexOf(q)>=0;});if(m)focusNode(m.id);}
  },300);
});

function focusNode(nid){
  net.focus(nid,{scale:1.5,animation:{duration:400,easingFunction:"easeInOutQuad"}});
  net.selectNodes([nid]);showNode(nid);
}

net.on("click",function(p){
  if(p.nodes.length>0)showNode(p.nodes[0]);
  else if(p.edges.length>0)showEdge(p.edges[0]);
});

function showNode(nid){
  var nd=allNodes.get(nid);if(!nd)return;
  var rels=[];
  allEdges.forEach(function(ed){
    if(ed.from===nid||ed.to===nid){
      var other=ed.from===nid?allNodes.get(ed.to):allNodes.get(ed.from);
      rels.push({edge:ed,other:other,dir:ed.from===nid?"out":"in"});
    }
  });
  document.getElementById("dhdr").textContent=nd.label;
  document.getElementById("detail").innerHTML=
    '<div class="ds"><div class="dl">类型</div><span class="tb" style="background:'+gc(nd._type)+'">'+nd._type+'</span></div>'+
    '<div class="ds"><div class="dl">描述</div><div class="dv">'+(nd._desc||"N/A")+'</div></div>'+
    '<div class="ds"><div class="dl">统计</div><div class="dv">关联度: '+nd._deg+' | 频次: '+nd._freq+'</div></div>'+
    '<div class="ds"><div class="dl">关系 ('+rels.length+')</div>'+
    rels.map(function(r){
      var txt=r.dir==="out"?nd.label+" → "+(r.other?r.other.label:"?"):(r.other?r.other.label:"?")+" → "+nd.label;
      var oid=r.other?r.other.id:0;
      return'<div class="ri" onclick="focusNode('+oid+')">'+txt+'<span class="rd">'+(r.edge._lb||"")+'</span></div>';
    }).join("")+'</div>';
}

function showEdge(eid){
  var ed=allEdges.get(eid);if(!ed)return;
  var f=allNodes.get(ed.from),t=allNodes.get(ed.to);
  document.getElementById("dhdr").textContent="关系";
  document.getElementById("detail").innerHTML=
    '<div class="ds" style="font-size:15px;"><span style="color:'+gc(f&&f._type)+'">'+(f?f.label:"?")+'</span> <span style="color:#5a7aff;">→</span> <span style="color:'+gc(t&&t._type)+'">'+(t?t.label:"?")+'</span></div>'+
    '<div class="ds"><div class="dl">描述</div><div class="dv">'+(ed._lb||"N/A")+'</div></div>'+
    '<div class="ds"><div class="dl">权重</div><div class="dv">'+ed._w+'</div></div>';
}
</script>
</body>
</html>'''


def _build_data_overview_html() -> str:
    """构建数据概览页面"""
    import base64
    import html as html_mod

    try:
        _init_graphrag()
        ent_df = _dataframes["entities"]
        rel_df = _dataframes["relationships"]
        comm_df = _dataframes["communities"]
        rpt_df = _dataframes["community_reports"]
        tu_df = _dataframes["text_units"]
    except Exception as e:
        return f"<p>数据加载失败: {html_mod.escape(str(e))}</p>"

    # 实体类型分布
    type_counts = graph_api._entities_df["type"].value_counts() if graph_api._entities_df is not None else ent_df.get("type", pd.Series()).value_counts()
    type_colors = {
        "PRODUCT": "#e74c3c", "ORGANIZATION": "#3498db", "CHEMICAL": "#2ecc71",
        "FORMULATION": "#f39c12", "CROP": "#27ae60", "PEST": "#e67e22",
        "ID": "#9b59b6", "DATE": "#8e44ad", "CONCEPT": "#1abc9c",
        "METHOD": "#e91e63", "OTHER": "#7f8c8d",
    }
    type_rows = ""
    max_count = type_counts.max() if len(type_counts) > 0 else 1
    for t, c in type_counts.items():
        color = type_colors.get(str(t), "#7f8c8d")
        pct = c / max_count * 100
        type_rows += f'<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:6px;"></span>{html_mod.escape(str(t))}</td><td>{c}</td><td><div style="background:#2a2a40;border-radius:3px;height:16px;width:100%;"><div style="background:{color};height:100%;width:{pct:.0f}%;border-radius:3px;"></div></div></td></tr>'

    # Top 20 实体
    top_entities = graph_api.search_entities("", limit=20)
    top_rows = ""
    for i, e in enumerate(top_entities, 1):
        color = type_colors.get(e["type"], "#7f8c8d")
        desc = html_mod.escape(e["description"][:80] + "..." if len(e["description"]) > 80 else e["description"])
        top_rows += f'<tr><td>{i}</td><td>{html_mod.escape(e["title"])}</td><td><span style="background:{color};color:#fff;padding:1px 8px;border-radius:10px;font-size:11px;">{html_mod.escape(e["type"])}</span></td><td>{e["degree"]}</td><td style="color:#888;font-size:12px;">{desc}</td></tr>'

    # 社区层级统计
    level_rows = ""
    rpt_comms = set(rpt_df["community"].unique())
    for lv in sorted(comm_df["level"].unique()):
        lv_comms = comm_df[comm_df["level"] == lv]
        lv_ids = set(lv_comms["community"].unique())
        n_total = len(lv_ids)
        n_rpt = len(lv_ids & rpt_comms)
        sizes = lv_comms.get("size", pd.Series())
        size_range = f"{int(sizes.min())}~{int(sizes.max())}" if len(sizes) > 0 else "-"
        level_rows += f'<tr><td>Level {int(lv)}</td><td>{n_total}</td><td>{n_rpt}</td><td>{n_total - n_rpt}</td><td>{size_range}</td></tr>'

    page = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#0f0f1a;color:#d8d8e8;line-height:1.7;padding:30px 40px;overflow-y:auto;}}
h1{{font-size:22px;color:#fff;margin-bottom:8px;}}
h2{{font-size:17px;color:#7aa2f7;margin:28px 0 12px;padding-left:10px;border-left:3px solid #7aa2f7;}}
.subtitle{{color:#888;font-size:13px;margin-bottom:24px;}}
.stats{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0;}}
.stat{{background:#1a1b26;border:1px solid #2a2a40;border-radius:8px;padding:10px 16px;text-align:center;min-width:100px;}}
.stat .num{{font-size:22px;color:#7aa2f7;font-weight:bold;}}
.stat .lbl{{font-size:11px;color:#888;margin-top:2px;}}
.card{{background:#1a1b26;border:1px solid #2a2a40;border-radius:10px;padding:20px 24px;margin:14px 0;}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px;}}
th{{background:#1e1e35;color:#9aa5ce;padding:8px 12px;text-align:left;border:1px solid #2a2a40;}}
td{{padding:8px 12px;border:1px solid #2a2a40;color:#c0c0d0;}}
tr:nth-child(even){{background:#151525;}}
</style>
</head>
<body>
<h1>数据概览</h1>
<p class="subtitle">知识图谱数据统计与分布</p>

<div class="stats">
  <div class="stat"><div class="num">{len(ent_df)}</div><div class="lbl">实体</div></div>
  <div class="stat"><div class="num">{len(rel_df)}</div><div class="lbl">关系</div></div>
  <div class="stat"><div class="num">{len(comm_df)}</div><div class="lbl">社区</div></div>
  <div class="stat"><div class="num">{len(rpt_df)}</div><div class="lbl">社区报告</div></div>
  <div class="stat"><div class="num">{len(tu_df)}</div><div class="lbl">文本片段</div></div>
  <div class="stat"><div class="num">{len(type_counts)}</div><div class="lbl">实体类型</div></div>
</div>

<h2>实体类型分布</h2>
<div class="card">
<table>
<tr><th>类型</th><th>数量</th><th>占比</th></tr>
{type_rows}
</table>
</div>

<h2>Top 20 高关联度实体</h2>
<div class="card">
<table>
<tr><th>#</th><th>实体名</th><th>类型</th><th>关联度</th><th>描述</th></tr>
{top_rows}
</table>
</div>

<h2>社区层级分布</h2>
<div class="card">
<table>
<tr><th>层级</th><th>社区数</th><th>有报告</th><th>无报告</th><th>Size 范围</th></tr>
{level_rows}
</table>
</div>

</body>
</html>'''

    b64 = base64.b64encode(page.encode("utf-8")).decode("ascii")
    return f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:800px;border:none;border-radius:8px;"></iframe>'


# ─── Gradio UI ──────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """构建 Gradio 界面"""
    # 预加载数据获取统计
    try:
        _init_graphrag()
        stats = graph_api.get_stats()
        stats_text = f"知识图谱已加载: {stats['entities']} 个实体, {stats['relationships']} 条关系"
    except Exception as e:
        stats_text = f"数据加载失败: {e}"

    with gr.Blocks(
        title="PestiGraph - 农药知识图谱",
    ) as app:
        gr.Markdown("# PestiGraph 农药知识图谱智能问答系统")
        gr.Markdown(stats_text)

        with gr.Tabs():
            # Tab 1: 智能问答
            with gr.Tab("智能问答"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=480, label="对话")
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="输入问题",
                                placeholder="请输入您的问题...",
                                scale=4,
                                lines=1,
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)
                            clear_btn = gr.Button("清空", variant="secondary", scale=1)
                        with gr.Accordion("查询参数", open=False):
                            with gr.Row():
                                mode_dd = gr.Dropdown(
                                    choices=["自动", "全局汇总", "精确检索", "关联分析", "基础检索"],
                                    value="自动",
                                    label="查询模式",
                                    info="默认自动路由，也可手动指定",
                                )
                                level_slider = gr.Slider(
                                    minimum=0, maximum=3, step=1, value=2,
                                    label="社区层级 (community_level)",
                                    info="越高覆盖越细，但越慢。0-1 适合快速概览，2 适合详细回答",
                                )
                                response_type_dd = gr.Dropdown(
                                    choices=list(RESPONSE_TYPE_MAP.keys()),
                                    value="详细多段落",
                                    label="回答格式 (response_type)",
                                    info="控制回答的详细程度和长度",
                                )
                        gr.Examples(
                            examples=[
                                "含有吡虫啉的农药有哪些？",
                                "防治稻飞虱用什么药？",
                                "草甘膦的安全间隔期是多少天？",
                                "常见的杀菌剂剂型有哪些？",
                                "三唑酮和哪些成分常搭配使用？",
                                "杀虫剂的主要类型有哪些？",
                                "甲维盐和高效氯氟氰菊酯的区别？",
                                "阿维菌素的登记企业有哪些？",
                            ],
                            inputs=msg_input,
                            examples_per_page=8,
                        )
                    with gr.Column(scale=1):
                        log_box = gr.HTML(
                            value=_get_log_html(),
                            elem_id="query-log",
                        )
                        log_timer = gr.Timer(value=1, active=False)

                def _user_submit(message, history, community_level, response_type_label, mode_override):
                    if not message.strip():
                        return "", history, _get_log_html()
                    history = history + [{"role": "user", "content": message}]
                    reply = chat(message, history, community_level, response_type_label, mode_override)
                    history = history + [{"role": "assistant", "content": reply}]
                    return "", history, _get_log_html()

                def _start_timer():
                    """查询开始时激活定时器并重置可视化"""
                    global _current_stage, _current_method, _query_elapsed
                    _query_log.clear()
                    _current_stage = "idle"
                    _query_elapsed = 0.0
                    return gr.Timer(active=True), _get_log_html()

                def _poll_log():
                    """定时刷新日志"""
                    return _get_log_html()

                # 点击/回车 → 先激活 timer，再执行查询，查询完停 timer
                for trigger in [send_btn.click, msg_input.submit]:
                    trigger(
                        fn=_start_timer,
                        inputs=[],
                        outputs=[log_timer, log_box],
                    ).then(
                        fn=_user_submit,
                        inputs=[msg_input, chatbot, level_slider, response_type_dd, mode_dd],
                        outputs=[msg_input, chatbot, log_box],
                    ).then(
                        fn=lambda: gr.Timer(active=False),
                        inputs=[],
                        outputs=[log_timer],
                    )

                log_timer.tick(
                    fn=_poll_log,
                    inputs=[],
                    outputs=[log_box],
                )

                def _clear_chat():
                    global _current_stage, _query_elapsed
                    _query_log.clear()
                    _current_stage = "idle"
                    _query_elapsed = 0.0
                    return [], _get_log_html()

                clear_btn.click(
                    fn=_clear_chat,
                    inputs=[],
                    outputs=[chatbot, log_box],
                )

            # Tab 2: 知识图谱浏览器
            with gr.Tab("知识图谱"):
                gr.HTML(value=_build_graph_viewer_html())

            # Tab 3: 数据概览
            with gr.Tab("数据概览"):
                gr.HTML(value=_build_data_overview_html())

        # 查询原理放在 Tabs 外面、页面底部，折叠展示
        with gr.Accordion("查询原理", open=False):
            gr.HTML(value=_build_architecture_html())

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
