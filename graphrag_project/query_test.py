"""
GraphRAG 交互式查询测试工具
支持 local / global / drift / basic 四种查询方式
"""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

METHODS = {
    "l": "local",
    "g": "global",
    "d": "drift",
    "b": "basic",
}

HELP_TEXT = """
╔══════════════════════════════════════════════════════╗
║          GraphRAG 交互式查询测试工具                ║
╠══════════════════════════════════════════════════════╣
║  输入问题后回车即可查询                              ║
║                                                      ║
║  切换查询方式（当前默认: local）:                     ║
║    /l  - Local Search  (向量+子图，适合具体问题)     ║
║    /g  - Global Search (社区报告，适合全局汇总)      ║
║    /d  - Drift Search  (漂移搜索)                    ║
║    /b  - Basic Search  (基础搜索)                    ║
║                                                      ║
║  其他命令:                                           ║
║    /help - 显示帮助                                  ║
║    /q    - 退出                                      ║
╚══════════════════════════════════════════════════════╝
"""


def query(method: str, question: str):
    cmd = [
        sys.executable, "-m", "graphrag", "query",
        "-r", ROOT,
        "-m", method,
        question,
    ]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            timeout=300,
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"[ERROR] {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("[ERROR] 查询超时（300秒）")


def main():
    method = None
    print(HELP_TEXT)

    while True:
        try:
            if method:
                prompt = f"[{method}] > "
            else:
                prompt = "[未选择模式，请先输入 /l /g /d /b] > "
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue

        if question == "/q":
            print("再见！")
            break
        elif question == "/help":
            print(HELP_TEXT)
            continue
        elif question.startswith("/"):
            key = question[1:]
            if key in METHODS:
                method = METHODS[key]
                print(f"已切换到 {method} 模式")
            else:
                print(f"未知命令: {question}，输入 /help 查看帮助")
            continue

        if not method:
            print("请先选择查询模式：/l (local) /g (global) /d (drift) /b (basic)")
            continue

        print(f"\n正在查询（{method}）...\n")
        query(method, question)
        print()


if __name__ == "__main__":
    main()
