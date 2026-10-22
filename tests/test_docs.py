"""课程质量测试：验证教材与引擎的一致性.

这是本套系统里最有价值的一组测试之一：它把 docs/ 下每篇课程的 Python
代码块**真的执行一遍**，确保教材里教的东西在当前引擎上确实存在、确实能跑。

教材与代码脱节是学习平台最常见的失败模式——读者照着抄却报错，比没有教材
更糟。这个测试让它不可能发生。
"""
from __future__ import annotations

import io
import re
import sys
import warnings
from pathlib import Path

import pytest

from qd.tutor import load_curriculum

# 需要跳过的代码块特征（示例里故意写的伪代码 / 需要外部数据 / 需要额外依赖）
SKIP_PATTERNS = [
    "pip install",
    "import akshare",
    "AkshareFeed()",          # 需要联网
    "akshare",                # 任何 akshare 调用都需要联网
    "TODO",
    "your_",                  # 占位符
    "mktcap_panel",           # 已知的教学占位
    "lightgbm",
    "import lightgbm",
    "# 省略",
]

# 已知安全的顶层导入，预置到执行环境
PRELUDE = """
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
"""


def collect_blocks():
    """返回 (课程id, 代码) 列表。"""
    c = load_curriculum()
    out = []
    for lesson in c:
        for code in lesson.code_blocks():
            out.append((lesson.id, code))
    return out


def test_curriculum_exists():
    c = load_curriculum()
    assert len(c) >= 60, "课程数量不足"


def test_every_lesson_well_formed():
    """每篇课程必须有：足够的字数、学习目标、核心要点、公式、代码块。"""
    c = load_curriculum()
    problems = []
    for l in c:
        if l.word_count < 900:
            problems.append(f"{l.id} 字数不足：{l.word_count}")
        if not l.objectives:
            problems.append(f"{l.id} 缺少 objectives")
        if not l.key_points:
            problems.append(f"{l.id} 缺少 key_points")
        if not l.code_blocks():
            problems.append(f"{l.id} 没有代码示例")
        if "$" not in l.body:
            problems.append(f"{l.id} 没有数学公式")
        if not l.category:
            problems.append(f"{l.id} 缺少 category")
    assert not problems, "\n".join(problems[:15])


def test_lesson_ids_unique():
    c = load_curriculum()
    ids = [l.id for l in c]
    assert len(ids) == len(set(ids)), "课程 id 有重复"


def test_prereq_ids_exist():
    """front matter 里的先修 id 必须真实存在（否则学习路径会断）。"""
    c = load_curriculum()
    known = {l.id for l in c}
    bad = []
    for l in c:
        for p in l.prereq:
            if p not in known:
                bad.append(f"{l.id} 的先修 {p} 不存在")
    assert not bad, "\n".join(bad[:10])


def test_internal_links_resolve():
    """正文里的相对链接必须指向真实文件（相对于该课程所在目录解析）。"""
    root = Path(__file__).resolve().parents[1] / "docs"
    c = load_curriculum()
    bad = []
    for l in c:
        # 课程文件所在目录（用 front matter 的 category）
        base = root / l.category
        for m in re.finditer(r"\]\(([^)]+\.md)\)", l.body):
            target = m.group(1)
            if target.startswith(("http", "#", "mailto:")):
                continue
            p = (base / target).resolve()
            if not p.exists():
                # 回退：把链接当成相对 docs 根目录
                p2 = (root / target).resolve()
                if not p2.exists():
                    bad.append(f"{l.id}: {target}")
    assert not bad, "死链：\n" + "\n".join(bad[:10])


def _patch_demo_cache():
    """把 generate_demo_market 换成带缓存的版本，避免每个代码块都重算一遍。

    课程示例里大量调用 ``generate_demo_market(n_symbols=..., seed=7)``，每次都
    重新模拟 2000 天 × 100 只股票要几百毫秒。测试里做缓存后整个检查从十几分钟
    降到几十秒。
    """
    import qd.data.synthetic as synth
    from functools import lru_cache

    orig = synth.generate_demo_market

    @lru_cache(maxsize=32)
    def cached(n_symbols=50, start="2018-01-01", end="2024-12-31", seed=7, **kw):
        # 测试用较小规模（25 只股票）保证速度，但**日期范围要保留完整**——
        # 有些课程会按 2023/2024 切片，截断数据会让它们失去样本。
        n = min(int(n_symbols), 25)
        return orig(n_symbols=n, start=start, end=end, seed=seed, **kw)

    synth.generate_demo_market = cached
    import qd.data as data_pkg

    if hasattr(data_pkg, "generate_demo_market"):
        data_pkg.generate_demo_market = cached
    return orig


def test_all_code_blocks_run():
    """逐篇执行课程里的 Python 代码块（跳过需要联网/外部依赖的）。

    跳过规则相当保守：只跳过明确需要联网（akshare）、需要额外依赖（lightgbm）
    或明确标注为占位的示例。其余**全部真实执行**——这是"教材与引擎不脱节"
    的硬保证。
    """
    blocks = collect_blocks()
    assert len(blocks) >= 50, f"代码块太少：{len(blocks)}"

    orig = _patch_demo_cache()
    failures = []
    executed = 0
    try:
        for lesson_id, code in blocks:
            if any(p in code for p in SKIP_PATTERNS):
                continue
            src = PRELUDE + "\n" + code
            buf = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = buf
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # 每个代码块独立执行（课程里的块大多是自包含的）
                    g = {"__name__": "__main__"}
                    exec(compile(src, f"<{lesson_id}>", "exec"), g)  # noqa: S102
                executed += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{lesson_id}: {type(exc).__name__}: {str(exc)[:200]}\n--- code ---\n{code[:300]}")
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
    finally:
        import qd.data.synthetic as synth

        synth.generate_demo_market = orig
        import qd.data as data_pkg

        data_pkg.generate_demo_market = orig

    total = sum(1 for _, c in blocks if not any(p in c for p in SKIP_PATTERNS))
    # 教材里的每一个可运行片段都必须真实跑通——这是"教材不脱节"的硬标准
    assert executed >= total, (
        f"只成功执行 {executed}/{total} 个代码块\n" + "\n\n".join(failures[:6])
    )
    assert not failures, f"{len(failures)} 个代码块执行失败：\n\n" + "\n\n".join(failures[:8])


def test_documented_apis_exist():
    """课程 front matter 里声明的 api 必须在引擎中真实存在。"""
    import importlib

    c = load_curriculum()
    bad = []
    for l in c:
        for api in l.api:
            parts = api.split(".")
            # 逐层尝试导入：qd.data.MarketData -> 导入 qd.data 后检查 MarketData
            ok = False
            for i in range(len(parts), 1, -1):
                mod_path = ".".join(parts[:i])
                try:
                    mod = importlib.import_module(mod_path)
                    rest = parts[i:]
                    obj = mod
                    for r in rest:
                        obj = getattr(obj, r)
                    ok = True
                    break
                except (ImportError, AttributeError):
                    continue
                except Exception:  # noqa: BLE001
                    break
            if not ok:
                bad.append(f"{l.id}: {api}")
    assert not bad, f"课程引用了不存在的 API：\n" + "\n".join(bad[:15])
