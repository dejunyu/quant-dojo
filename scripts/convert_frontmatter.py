#!/usr/bin/env python3
"""把课程从 YAML front matter 迁移成 HTML 注释元信息（并可选把
学习目标/练习提升为正文二级标题）。

为什么这么做
------------
GitHub 不认 front matter，会把 YAML 里的每个 ``key: value`` 渲染成一张表格。
结果是一篇 130 行的教程，开头先出现 8 张「id / title / category / level /
order / prereq / tags / objectives ...」的表格，读者第一眼看到的是
``id: 01-01``，正文被挤到第八张表之后。

HTML 注释 ``<!-- ... -->`` 在 GitHub 上完全不可见，而 ``qd.tutor`` 照样能
解析出全部字段，所以迁移后：GitHub 上干净，平台上功能不变。

用法
----
    python scripts/convert_frontmatter.py --dry-run docs/01-入门
    python scripts/convert_frontmatter.py docs/01-入门
    python scripts/convert_frontmatter.py docs            # 全量
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# 写进正文（GitHub 读者能看见）的字段：这些是「内容」，不是「元数据」。
INLINE_SECTIONS = ("objectives", "key_points", "exercises")

SECTION_TITLES = {
    "objectives": "学完这一篇，你能做到",
    "key_points": "本篇要点",
    "exercises": "动手练习",
}

# 哪些章节插在正文之前（导读），哪些放在正文之后（练习）
HEAD_SECTIONS = ("objectives", "key_points")
TAIL_SECTIONS = ("exercises",)

# 留在注释块里的字段（平台导航 / 搜索 / 校验用，读者不需要看）
# 输出顺序固定，避免每次 diff 抖动
FIELD_ORDER = [
    "id", "title", "category", "level", "order",
    "prereq", "tags", "key_points", "api",
]


def split_front_matter(text: str) -> tuple[dict, str] | None:
    """返回 (元信息 dict, 正文)；不是旧格式则返回 None。"""
    s = text.lstrip("\ufeff")
    if not s.startswith("---"):
        return None
    parts = s.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(fm, dict) or "id" not in fm:
        return None
    return fm, parts[2].lstrip("\n")


def _dump_yaml(fm: dict) -> str:
    """按固定字段顺序 dump，风格紧凑（列表优先用行内写法）。"""
    ordered = {k: fm[k] for k in FIELD_ORDER if k in fm}
    for k, v in fm.items():          # 保留未知字段，避免信息丢失
        ordered.setdefault(k, v)

    lines = []
    for key, value in ordered.items():
        if isinstance(value, list):
            if not value:
                lines.append(f"{key}: []")
            elif all(isinstance(x, str) and "\n" not in x for x in value):
                # 短列表用行内写法，长文本用缩进块
                inline = "[" + ", ".join(x for x in value) + "]"
                if len(inline) <= 100 and not any("," in x for x in value):
                    lines.append(f"{key}: {inline}")
                else:
                    lines.append(f"{key}:")
                    for x in value:
                        lines.append(f"  - {x}")
            else:
                lines.append(f"{key}:")
                for x in value:
                    lines.append(f"  - {x}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_body(body: str, fm: dict) -> str:
    """把 objectives / exercises 提升为正文二级标题。"""
    out = body

    def block(key):
        items = fm.get(key) or []
        if not items:
            return ""
        return "\n".join([f"## {SECTION_TITLES[key]}", ""]
                         + [f"- {x}" for x in items]) + "\n\n"

    head = "".join(block(k) for k in HEAD_SECTIONS)
    if head:
        m = re.search(r"^#\s+.*$", out, re.M)
        if m:
            out = out[:m.end()] + "\n\n" + head + out[m.end():].lstrip("\n")
        else:
            out = head + out

    tail = "".join(block(k) for k in TAIL_SECTIONS)
    if tail:
        out = out.rstrip("\n") + "\n\n" + tail

    return out


def convert_file(path: Path, dry_run: bool = False) -> str | None:
    """返回状态描述；无需处理时返回 None。"""
    text = path.read_text(encoding="utf-8")
    split = split_front_matter(text)
    if split is None:
        return None                      # 已是新格式或不是课程文件
    fm, body = split

    new_meta = {k: v for k, v in fm.items() if k not in INLINE_SECTIONS}
    header = "<!--\n" + _dump_yaml(new_meta) + "\n-->\n\n"
    new_text = header + build_body(body, fm).rstrip("\n") + "\n"

    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return f"{'[dry] ' if dry_run else ''}{path}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="课程元信息格式迁移")
    ap.add_argument("targets", nargs="+", help="文件或目录")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    files: list[Path] = []
    for t in args.targets:
        p = Path(t)
        if not p.is_absolute():
            p = (root / t) if (root / t).exists() else p
        files.extend(sorted(p.rglob("*.md")) if p.is_dir() else [p])

    changed = 0
    for f in files:
        status = convert_file(f, dry_run=args.dry_run)
        if status:
            print(status)
            changed += 1

    print(f"\n共处理 {changed} / {len(files)} 个文件"
          f"{'（dry-run，未写入）' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
