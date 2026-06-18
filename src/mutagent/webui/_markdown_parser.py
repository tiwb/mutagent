"""增量 Markdown 解析器 — 按标题和 fenced code block 切分文本。

流式场景下逐行消费，状态机跟踪当前是否在 code fence 内。
产出 Section 列表，通过 ``visible_items()`` 扁平展开为 ChatItem 列表，
折叠状态由后端管理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ._messages import CodeBlockItem, MarkdownItem, SectionHeadingItem, ChatItem

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^(`{3,})(.*)$")


@dataclass
class _Section:
    """解析器内部结构，记录标题层级关系用于折叠逻辑。"""
    id: str
    title: str
    level: int
    parent_id: str | None = None
    collapsed: bool = False
    chunks: list[MarkdownItem | CodeBlockItem] = field(default_factory=lambda: [])


class IncrementalMarkdownParser:
    """流式增量 markdown 解析器。

    逐行消费，状态机跟踪当前是否在 code fence 内。
    产出 Section 列表，通过 ``visible_items()`` 扁平展开为 ChatItem 列表。

    用法::

        parser = IncrementalMarkdownParser(prefix="turn-1-")
        # 流式
        items = parser.feed("## 标题\\n")
        items = parser.feed("正文内容...")
        items = parser.finalize()
        # 完整文本
        items = parser.parse_complete(full_text)
    """

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix
        self._sections: list[_Section] = []
        self._pre_chunks: list[MarkdownItem | CodeBlockItem] = []
        self._text_buffer: list[str] = []
        self._line_buffer: str = ""  # 尚未完成的行
        self._in_fence: bool = False
        self._fence_count: int = 0
        self._fence_lang: str = ""
        self._fence_buffer: list[str] = []
        self._section_counter: int = 0
        self._chunk_counter: int = 0

    # ── 公共 API ──────────────────────────────────────────────────

    def feed(self, new_text: str) -> list[ChatItem]:
        """追加文本，返回当前可见 ChatItem 列表。

        逐行消费，处理标题拆分和 fenced code block 识别。
        """
        if not new_text:
            return self.visible_items()
        self._line_buffer += new_text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._process_line(line)
        return self.visible_items()

    def finalize(self) -> list[ChatItem]:
        """流式结束，产出最终列表。

        处理末闭合行、未完成的 fence、尾部文本。
        """
        if self._line_buffer:
            self._process_line(self._line_buffer)
            self._line_buffer = ""
        self._flush_text()
        if self._in_fence:
            # 未闭合 fence — 降级为普通文本
            self._text_buffer.append("`" * self._fence_count + self._fence_lang)
            self._text_buffer.extend(self._fence_buffer)
            self._in_fence = False
            self._fence_count = 0
            self._fence_lang = ""
            self._fence_buffer = []
            self._flush_text()
        return self.visible_items()

    def parse_complete(self, text: str) -> list[ChatItem]:
        """完整文本一次解析（用于 rebuild 非流式路径）。"""
        if text:
            for line in text.split("\n"):
                self._process_line(line)
        return self.finalize()

    def visible_items(self) -> list[ChatItem]:
        """按 section 列表扁平展开可见 item。

        折叠的 section 其子孙（level 更大的后续 section）不出现。
        使用 ``skip_above`` 逐一遍历，O(n)。
        """
        result: list[ChatItem] = list(self._pre_chunks)
        skip_above: int | None = None
        for s in self._sections:
            if skip_above is not None and s.level > skip_above:
                continue
            skip_above = None
            result.append(SectionHeadingItem(
                id=s.id,
                kind="section_heading",
                text=s.title,
                level=s.level,
                collapsed=s.collapsed,
            ))
            result.extend(s.chunks)
            if s.collapsed:
                skip_above = s.level
        return result

    def toggle_collapse(self, section_id: str) -> None:
        """切换指定 section 的折叠状态。"""
        for s in self._sections:
            if s.id == section_id:
                s.collapsed = not s.collapsed
                return

    # ── 内部方法 ──────────────────────────────────────────────────

    def _process_line(self, line: str) -> None:
        """处理单行文本。"""
        if self._in_fence:
            self._process_fence_line(line)
        else:
            self._process_normal_line(line)

    def _process_normal_line(self, line: str) -> None:
        """处理非 fence 内的行。"""
        hm = _HEADING_RE.match(line)
        if hm:
            self._flush_text()
            level = len(hm.group(1))
            title = hm.group(2).strip()
            self._section_counter += 1
            sid = f"{self._prefix}section-{self._section_counter}"
            parent_id = self._find_parent_id(level)
            section = _Section(
                id=sid, title=title, level=level, parent_id=parent_id
            )
            self._sections.append(section)
            return
        fm = _FENCE_RE.match(line)
        if fm:
            self._flush_text()
            self._in_fence = True
            self._fence_count = len(fm.group(1))
            self._fence_lang = fm.group(2).strip()
            self._fence_buffer = []
            return
        self._text_buffer.append(line)

    def _process_fence_line(self, line: str) -> None:
        """处理 fence 内的行。

        关闭条件（遵循 CommonMark 规范）：
        - 反引号数量 ≥ 开启 fence 的数量
        - 反引号后无信息字符串（仅允许空白）
        """
        fm = _FENCE_RE.match(line)
        if fm:
            closing_count = len(fm.group(1))
            closing_info = fm.group(2).strip()
            if closing_count >= self._fence_count and not closing_info:
                # fence 关闭
                self._chunk_counter += 1
                cid = f"{self._prefix}code-{self._chunk_counter}"
                code = "\n".join(self._fence_buffer)
                chunk = CodeBlockItem(
                    id=cid, kind="code_block", code=code,
                    language=self._fence_lang, fence_count=self._fence_count,
                )
                self._append_chunk(chunk)
                self._in_fence = False
                self._fence_count = 0
                self._fence_lang = ""
                self._fence_buffer = []
            else:
                self._fence_buffer.append(line)
        else:
            self._fence_buffer.append(line)

    def _flush_text(self) -> None:
        """将 _text_buffer 产出为一个 MarkdownItem。

        去掉头尾空行，纯空白内容跳过不产出。
        """
        if not self._text_buffer:
            return
        text = "\n".join(self._text_buffer).strip()
        if not text:
            self._text_buffer = []
            return
        self._chunk_counter += 1
        cid = f"{self._prefix}md-{self._chunk_counter}"
        chunk = MarkdownItem(id=cid, kind="markdown", text=text)
        self._append_chunk(chunk)
        self._text_buffer = []

    def _append_chunk(self, chunk: MarkdownItem | CodeBlockItem) -> None:
        """将 chunk 追加到当前 section 或 pre_chunks。"""
        if self._sections:
            self._sections[-1].chunks.append(chunk)
        else:
            self._pre_chunks.append(chunk)

    def _find_parent_id(self, level: int) -> str | None:
        """找到 level 小于当前 level 的最近一个 section 的 id。"""
        for s in reversed(self._sections):
            if s.level < level:
                return s.id
        return None
