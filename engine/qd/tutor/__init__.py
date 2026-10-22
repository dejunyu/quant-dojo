"""课程体系：扫描 docs 下的 Markdown 课程，构建结构化知识树."""
from .curriculum import Curriculum, Lesson, load_curriculum

__all__ = ["Curriculum", "Lesson", "load_curriculum"]
