# feature-mcp-schema-help-display（iter1：验收反馈与修正）

**状态**：✅ 已完成
**日期**：2026-05-13
**类型**：功能迭代
**来源**：`feature-mcp-schema-help-display.md`

## 验收发现的问题

2026-05-13 在聚合 server（mutagent pysandbox --port 8700）上完成手工验收，
发现以下三个问题。问题 1 和问题 2 都属于「让 `help()` 在用户眼里对齐」这个
主目标的两半（docstring 对齐 + signature 对齐），本次迭代一并修复。问题 3
不做处理。

### 问题 1: description 与约束后缀视觉混淆

**现象**：当上游 description 不以句号结尾时，描述与约束后缀直接空格拼接，人眼
难以辨别哪些文字来自上游、哪些是我们的翻译。

```
action: string (required) — Operation to perform Allowed: list | new | close | select.
                              ↑ 上游 description         ↑ 我们的约束后缀（混在一起）
```

即使 description 以句号结尾（如 `"Defaults to \"info\"."`），边界也仅靠一个
空格外加首字母大写区分，不够明确。

**方案对比**：

| 方案 | 效果 | 优点 | 缺点 |
|------|------|------|------|
| A. 补句号 | `...perform. Allowed: ...` | 改动最小 | 修改了上游文本；区分度仍不高 |
| B. 独立缩进行 | 见下 | 最清晰，人眼秒区分 | 多占一行 |
| C. 括号包裹 | `...perform (Allowed: ...)` | 紧凑 | 容易和参数名/值括号混淆 |
| D. 方括号前缀 | `...perform [Allowed: ...]` | 有区分度 | 像可选语义，不够自然 |

**采用方案 B：独立缩进行**。约束后缀单独一行、缩进 8 空格（基础 4 + 续行 4，
与描述段落对齐）。

```
playwright.browser_tabs(action: str, index: float, url: str)

List, create, close, or select a browser tab.

Args:
    action: string (required) — Operation to perform
        Allowed: list | new | close | select.
    index: number — Tab index, used for close/select.
    url: string — URL to navigate to in the new tab.
```

多约束并存时同一行内空格分隔：

```
    paths: array — Local file paths.
        Items: string. Items count: 1..10. Items must be unique.
```

**续行缩进为何选 8 空格（基础 4 + 续行 4）**：

- **Google-style docstring 解析器约束**：参数项本身缩进 4，续行必须严格 > 4
  才不被 Napoleon/Sphinx 解析为「上一条结束」。5/6/8 都合法，但主流惯例是
  「深一级 = 再 +4」。
- **一致性**：基础缩进是 4，续行再 +4 是最无歧义的选择，读者不需要数列数。
- **子块语义**：suffix 是参数描述的附属信息、且来源不同（我们生成 vs 上游
  提供），视觉上成块下沉更能表达「这是附加说明」。
- **为未来扩展留空间**：若将来 Args 内出现嵌套说明（如 union 类型展开），
  8 空格留出了 6 空格等中间层级。

2 空格缩进（总 6）虽然也合法且更紧凑，但相对参数项 4 只深 2，续行夹角太小，
在等宽字体下和 em-dash 后的描述列层级感不足。

**边界情况（四分支规则）**：

| `pdesc` | `suffix` | 产出格式 |
|---------|----------|---------|
| 非空 | 非空 | 两行：`    name: type{req} — {pdesc}` + `        {suffix}` |
| 非空 | 空 | 单行：`    name: type{req} — {pdesc}` |
| 空 | 非空 | 单行：`    name: type{req} — {suffix}`（不要空悬 em-dash + 换行） |
| 空 | 空 | 单行：`    name: type{req}`（不加 em-dash） |

注意第四种情况（无描述无约束）当前实现也会留下 `— ` 空尾，顺手一并修净。

**实施影响**：

- `format_param_description_suffix` 本身不变——它仍是纯函数，返回同格式字符串
- 变的是 `_adapter_mcp.py` 中 `_make_tool_func` 的 docstring 组装逻辑：
  当前 `{pdesc} {suffix}` 同行拼接 → 改为四分支规则输出
- 集成测试断言需同步更新（格式变了，但约束内容不变）
- `help()` 输出中的 Args 区域视觉更清晰，但对 `ParamDescr.doc` 字符串做
  正则抽取的消费者可能有影响（目前已知无此类消费者）

### 问题 2: namespace sharing 代理丢失 `_MISSING` sentinel

**现象**：同一个 MCP tool 在不同路径下签名不同。

| 路径 | `browser_file_upload` 签名 |
|------|--------------------------|
| 直连（8765） | `paths: list = ...` ✅ |
| 共享 namespace（8700） | `paths: list` ❌ 缺 `= ...` |

optional-no-default 参数的 `= ...` 在 namespace sharing 代理层丢失，用户
无法从签名区分「有默认值」和「可选但无默认」。

**根因分析**：`_MISSING` sentinel 是进程内 Python singleton，跨 sandbox 边界
序列化时身份丢失。具体数据流：

1. **直连侧**（`_adapter_mcp.py:_make_tool_func`）：`build_signature` 正确
   构造 `Parameter(default=_MISSING)`，`__signature__` 显示 `= ...`（因
   `_MissingSentinel.__repr__` 返回 `"..."`）。✅

2. **共享侧服务端**（`share.py:_describe_function`）：
   ```python
   if _default_is_json_safe(p.default):    # json.dumps(_MISSING) 抛 TypeError
       entry["default"] = p.default
   else:
       entry["default_repr"] = repr(p.default)   # _MISSING 走这，schema 无 default 字段
   ```
   → schema 只留字符串 `"..."`，`_MISSING` 身份完全丢失。

3. **共享侧客户端**（`_signature.py:build_signature`）：
   ```python
   has_default = "default" in spec    # default_repr 不被识别
   ```
   → `has_default=False` → 参数被当必填 → 签名缺 `= ...`。

**修复方案：schema-level 标记位**（additive 向前/向后兼容）

服务端 `_describe_function` 增加 `_MISSING` 分支：
```python
if p.default is _MISSING:
    entry["default_missing"] = True
elif _default_is_json_safe(p.default):
    entry["default"] = p.default
else:
    entry["default_repr"] = repr(p.default)
```

客户端 `build_signature` 识别新字段：
```python
if spec.get("default_missing"):
    default = _MISSING
    has_default = True
elif "default" in spec:
    default = spec["default"]
    has_default = True
else:
    default = inspect.Parameter.empty
    has_default = False
```

**为何不用其他方案**：

- **从 `default_repr="..."` 反推** `_MISSING`：不可靠。合法 Python 代码可能
  真有参数默认值就是 `Ellipsis`（`def f(x=...)` 是合法签名），字符串相等
  不保证身份。必须显式标记。
- **让 `_default_is_json_safe` 接受 sentinel**：`_MISSING` 不是 JSON 原生值，
  强行序列化会污染其他消费者（前端展示层、其他 RPC 场景）。schema 层加
  标记位是最小侵入。

**兼容性**（additive，完全向前/向后兼容）：

| 场景 | 行为 |
|------|------|
| 新 server + 新 client | ✅ 正确显示 `= ...` |
| 新 server + 老 client | 老 client 忽略 `default_missing` 未知字段 → 参数被当必填（现状行为，降级安全）|
| 老 server + 新 client | 老 server 发 `default_repr="..."` → 新 client 不识别 → 参数被当必填（现状行为）|
| 老 + 老 | 现状 |

### 问题 3: 上游 description 自带 default 信息导致双写

**现象**：上游 MCP server 的 `description` 字段中已包含默认值说明，签名中也
展示了 `= value`，形成双写。

```
Args:
    level: string — ... Defaults to "info". Allowed: error | warning | info | debug.
                     ↑ 上游自带的 default 描述    ↑ 我们的约束后缀

签名: browser_console_messages(level: str = 'info', ...)
                                 ↑ Python 签名中的 default
```

**处理**：不做处理。上游 description 内容不受我们控制，且签名层的 default 是
机器可消费的精确值（`= 'info'`），description 层的 `Defaults to "info"` 是
自然语言补充——两者角色不同，不视为冗余 bug。我们的 spec 已经明确「docstring
不重复签名已表达的信息」，自身没有往 docstring 追加 default，问题在上游。
尝试正则剥离上游文本风险高收益低。

---

## 实施步骤

### Phase 4：方案 B — 约束后缀独立缩进行（docstring 对齐）

- [x] 修改 `_adapter_mcp.py` 中 `_make_tool_func` 的 docstring 组装逻辑
      （查找 `doc_lines.append(f"    {pname}` 处），从 `{pdesc} {suffix}`
      同行拼接改为四分支规则输出
- [x] 更新 Phase 2a 集成测试断言，匹配新格式（两行 + 8 空格缩进）
- [x] 回归全部相关测试
- [x] 手工验收：8700 + 8765 两端 `help()` 输出 Args 区域确认清晰可辨
      （2026-05-13：直连 8765 和共享 8700 的 `__doc__` / `help()` 输出完全一致，
      约束后缀正确独立缩进 8 空格）

### Phase 5：修复 `_MISSING` 跨 sharing 身份丢失（signature 对齐）

- [x] 修改 `share.py:_describe_function`：import `_MISSING`，新增
      `default_missing=True` 分支（放在 `_default_is_json_safe` 检查之前）
- [x] 修改 `_signature.py:build_signature`：识别 `default_missing` 字段，
      还原为 `default=_MISSING`
- [x] 新增测试：sharing 链路 optional-no-default 参数签名断言
      （服务端发 `default_missing`、客户端 `__signature__` 含 `= ...`）
- [x] 手工验收：8700 和 8765 两端同一 tool 的 `help()` 签名完全一致
      （2026-05-13：`browser_tabs` / `browser_console_messages` /
      `browser_file_upload` / `browser_drop` 四 tool 的 `__signature__` 和
      `help()` 输出在两端完全一致，`= ...` 正确保留）

Phase 4 和 Phase 5 正交独立，不互相阻塞，按顺序实施即可。

### 遗留

- 问题 3（上游 description 自带 default 双写）不做处理

## 关键参考

- 主 spec：`feature-mcp-schema-help-display.md`
- `src/mutagent/sandbox/_adapter_mcp.py` — `_make_tool_func` 中 docstring
  组装段（查找 `doc_lines.append(f"    {pname}`），Phase 4 改动点
- `src/mutagent/sandbox/_signature.py` — `format_param_description_suffix`
  （Phase 4 不动）；`build_signature` 中 `has_default = "default" in spec`
  一行（Phase 5 改动点）
- `src/mutagent/sandbox/share.py` — `_describe_function` 中
  `_default_is_json_safe(p.default)` 分支（Phase 5 改动点）

## 最终验收（2026-05-13）

### 自动化验证

```bash
python -m pytest tests/test_adapter_mcp.py tests/test_signature_build.py tests/test_pysandbox_sharing.py -q
# 148 passed in 0.48s ✅
```

### 手工验收

| 验收项 | 8765 直连 | 8700 共享 | 结论 |
|--------|----------|----------|------|
| Phase 4: docstring 约束后缀独立缩进 | ✅ `Allowed: …` / `Items: …` / `Extra values: …` 均独立行、8 空格缩进 | ✅ 与 8765 完全一致 | 通过 |
| Phase 5: `= …` 签名跨 sharing 保留 | ✅ `index: float = …` / `paths: list = …` | ✅ 与 8765 完全一致 | 通过 |

验收使用的 tool：`browser_tabs`、`browser_console_messages`、`browser_file_upload`、
`browser_drop`（覆盖 enum、range、items、additionalProperties 等全量约束类型）。

两端 `__doc__` 和 `__signature__` 通过 `repr()` 严格比对，
完全一致（之前观察到的差异是终端 capture 换行 artifact）。
