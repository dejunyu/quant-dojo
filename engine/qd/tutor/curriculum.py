"""课程加载器.

课程就是 ``docs/`` 下的 Markdown 文件，每个文件开头有 YAML front matter
（id / title / category / level / prereq / objectives / key_points / exercises / api）。
平台在启动时扫描一次，构建出知识树 + 索引 + 搜索，供 Web UI 与 CLI 使用。

这样做的好处：写课程 = 写 Markdown，不需要改任何代码。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["Lesson", "Curriculum", "load_curriculum", "DEFAULT_DOCS"]

# 文件位于 <root>/engine/qd/tutor/curriculum.py，课程在 <root>/docs
DEFAULT_DOCS = Path(__file__).resolve().parents[3] / "docs"


@dataclass
class Lesson:
    """一篇课程."""
    id: str
    title: str
    category: str
    level: str = "入门"
    order: int = 999
    prereq: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    exercises: list[str] = field(default_factory=list)
    api: list[str] = field(default_factory=list)
    body: str = ""
    path: str = ""

    @property
    def word_count(self) -> int:
        """正文中文字数（粗略：统计中日韩字符）。"""
        return len(re.findall(r"[\u4e00-\u9fff]", self.body))

    @property
    def reading_minutes(self) -> int:
        return max(1, self.word_count // 400)

    def code_blocks(self) -> list[str]:
        return re.findall(r"```python\n(.*?)```", self.body, flags=re.S)

    def to_dict(self, with_body: bool = False) -> dict:
        d = {
            "id": self.id, "title": self.title, "category": self.category,
            "level": self.level, "order": self.order, "prereq": self.prereq,
            "tags": self.tags, "objectives": self.objectives,
            "key_points": self.key_points, "exercises": self.exercises,
            "api": self.api, "word_count": self.word_count,
            "reading_minutes": self.reading_minutes, "path": self.path,
        }
        if with_body:
            d["body"] = self.body
        return d


class Curriculum:
    """课程集合."""

    def __init__(self, lessons: list[Lesson]):
        self.lessons = sorted(lessons, key=lambda l: (l.category, l.order, l.id))
        self._by_id = {l.id: l for l in self.lessons}

    # ------------------------------------------------------------ 查询
    def __len__(self) -> int:
        return len(self.lessons)

    def __iter__(self):
        return iter(self.lessons)

    def __getitem__(self, id_: str) -> Lesson:
        return self._by_id[id_]

    def get(self, id_: str) -> Lesson | None:
        return self._by_id.get(id_)

    def categories(self) -> list[dict]:
        """返回分类目录（含每类的课程数与总字数）。"""
        cats: dict[str, dict] = {}
        for l in self.lessons:
            c = cats.setdefault(l.category, {
                "id": l.category, "title": l.category, "n_lessons": 0,
                "word_count": 0, "lessons": [],
            })
            c["n_lessons"] += 1
            c["word_count"] += l.word_count
            c["lessons"].append(l.to_dict())
        return sorted(cats.values(), key=lambda c: c["id"])

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """全文检索（标题 / 标签 / 要点 / 正文）。"""
        q = query.strip().lower()
        if not q:
            return []
        scored = []
        for l in self.lessons:
            score = 0
            if q in l.title.lower():
                score += 10
            if any(q in t.lower() for t in l.tags):
                score += 5
            if any(q in k.lower() for k in l.key_points):
                score += 3
            if q in l.body.lower():
                score += 1
            if score:
                scored.append((score, l))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [l.to_dict() for _, l in scored[:limit]]

    def learning_path(self) -> list[dict]:
        """按先修关系给出推荐学习顺序（拓扑排序，环则按 id 兜底）。"""
        remaining = {l.id: set(l.prereq) & set(self._by_id) for l in self.lessons}
        done: list[str] = []
        while remaining:
            ready = [i for i, p in remaining.items() if not (p - set(done))]
            if not ready:  # 有环，按 id 取一个打破
                ready = [sorted(remaining)[0]]
            ready.sort()
            for i in ready:
                done.append(i)
                remaining.pop(i)
        return [self._by_id[i].to_dict() for i in done]

    def stats(self) -> dict:
        levels: dict[str, int] = {}
        for l in self.lessons:
            levels[l.level] = levels.get(l.level, 0) + 1
        return {
            "n_lessons": len(self.lessons),
            "n_categories": len({l.category for l in self.lessons}),
            "total_words": sum(l.word_count for l in self.lessons),
            "total_reading_minutes": sum(l.reading_minutes for l in self.lessons),
            "levels": levels,
            "n_code_blocks": sum(len(l.code_blocks()) for l in self.lessons),
        }

    def api_index(self) -> dict[str, list[str]]:
        """API -> 出现在哪些课程里（便于「边学边查」）。"""
        idx: dict[str, list[str]] = {}
        for l in self.lessons:
            for a in l.api:
                idx.setdefault(a, []).append(l.id)
        return idx


def load_curriculum(docs_dir=None) -> Curriculum:
    """扫描 docs 目录，加载全部课程。"""
    root = Path(docs_dir).expanduser() if docs_dir else DEFAULT_DOCS
    if not root.exists():
        return Curriculum([])

    lessons: list[Lesson] = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        lesson = _parse_lesson(text, str(path.relative_to(root)))
        if lesson:
            lessons.append(lesson)
    return Curriculum(lessons)


def _parse_lesson(text: str, rel_path: str) -> Lesson | None:
    """解析带 YAML front matter 的 Markdown。"""
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    raw_fm, body = parts[1], parts[2]
    try:
        fm = yaml.safe_load(raw_fm) or {}
    except yaml.YAMLError:
        fm = {}
    if not isinstance(fm, dict) or "id" not in fm:
        return None

    def lst(key):
        v = fm.get(key, [])
        return [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])

    return Lesson(
        id=str(fm.get("id")),
        title=str(fm.get("title", "")),
        category=str(fm.get("category", "未分类")),
        level=str(fm.get("level", "入门")),
        order=int(fm.get("order", 999) or 999),
        prereq=lst("prereq"),
        tags=lst("tags"),
        objectives=lst("objectives"),
        key_points=lst("key_points"),
        exercises=lst("exercises"),
        api=lst("api"),
        body=body.strip(),
        path=rel_path,
    )
