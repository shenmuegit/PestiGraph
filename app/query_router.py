"""
智能路由：根据问题类型自动选择 GraphRAG 查询模式
零成本、零延迟，纯关键词规则
"""
import re


def route(question: str) -> tuple[str, str]:
    """根据问题内容自动选择查询模式，返回 (模式, 匹配关键词)"""
    q = question.strip()

    # Global: 全局汇总、统计、排名类
    _global_kw = r"有哪些|总结|排名|最多|最少|统计|概况|分布|常见|多少种|占比|趋势|全部|所有|列举"
    m = re.search(_global_kw, q)
    if m:
        return "global", m.group()

    # Drift: 跨领域关联、比较类
    _drift_kw = r"关系|搭配|复配|组合|关联|比较|区别|异同|替代|相似|混用"
    m = re.search(_drift_kw, q)
    if m:
        return "drift", m.group()

    # Local: 默认，具体实体/产品/成分查询
    return "local", ""


# 模式说明，用于 UI 展示
MODE_LABELS = {
    "local": "精确检索",
    "global": "全局汇总",
    "drift": "关联分析",
    "basic": "基础检索",
}
