# 沙箱安全模型

mutagent 的沙箱执行环境（pysandbox）为 agent 提供受控的 Python 代码执行能力。本文档定义沙箱的安全模型——封闭性定理、威胁分析、实现机制和 builtin 安全分析。

**实现状态**：当前 mutagent 实现是简化版本（仅 `__builtins__` 白名单 + AST 禁止 import），未实现属性守卫、ScriptEnv 冻结和 `_safe_build_class`。本文档描述完整安全模型，作为未来实现的设计参考。简化版本有意不防御 agent 逃逸——agent 是 LLM，不是对抗性攻击者。

**依赖**：白名单由 TypeSpec 声明定义，见 [mutobj type-spec.md](../../../mutobj/docs/design/type-spec.md)。

## 安全子集语言

定义一个 Python 语言子集，满足两个性质：

1. **最大兼容性**：尽可能接近完整 Python，只排除破坏封闭性的语法
2. **可证明封闭**：给定一组类型声明（TypeSpec），子集代码不可能访问到声明范围之外的对象

### 封闭性的形式化定义

**定义**：一段代码是"封闭的"，当且仅当它执行过程中访问到的所有对象，要么是局部创建的，要么是宿主显式注入的，要么是通过声明范围内的操作从上述对象派生的。

**等价表述**：不存在一条执行路径，使得代码获取到"宿主未注入、且不可从已注入对象通过声明操作派生"的对象引用。

### 封闭性的威胁模型

Python 中获取"超出当前作用域"的对象引用的途径：

| 途径 | 机制 | 封堵方式 |
|------|------|---------|
| 模块导入 | `import` 语句 | AST 层面禁止 |
| 隐式内置 | `__builtins__` 提供 `eval`、`exec`、`__import__`、`getattr` 等 | 置空 `__builtins__` |
| 对象图遍历 | 沿 `.attr` 链从安全对象走到不安全对象 | **属性访问白名单** |

前两者是"符号注入"问题，通过控制执行环境解决。第三者是核心挑战。

### 对象图遍历的唯一语法通道

Python 中对对象执行操作的方式：

| 操作 | 语法 | 是否产生逃逸 | 安全性依据 |
|------|------|-------------|-----------|
| 属性访问 | `obj.attr` | **是** — 可返回任意对象 | 需要白名单守卫 |
| 运算符 | `a + b` | 否 | 不构成逃逸——即使返回未知类型的对象，后续属性访问仍受守卫约束。基础类型的运算符返回类型由 CPython 内置实现保证（int+int→int）；Declaration 子类的 dunder 返回类型由 `@impl` 作者负责，不在沙箱的安全保证范围内 |
| 下标 | `obj[key]` | 否 | 不构成逃逸——`__getitem__` 是 dunder 调用，返回值的后续属性访问仍受守卫约束 |
| 迭代 | `for x in obj` | 否 | 不构成逃逸——`__iter__`/`__next__` 是 dunder 调用，产出值的后续属性访问仍受守卫约束 |
| 函数调用 | `f(args)` | 否 | `f` 的来源已受控（注入或局部定义）。`f` 的返回值可以是任意类型，但返回值后续的属性访问仍受守卫约束，不构成逃逸 |
| 隐式 dunder | `__enter__`、`__format__` 等 | 否 | 由语句触发，对象来源已受控 |

**属性访问是唯一能产生逃逸的语法通道。** 其他操作即使返回了未知类型的对象，只要该对象后续的属性访问经过白名单检查，就不构成逃逸——对象本身无害，访问其属性才可能有害。

### 封闭性定理

**定理**：如果一段 Python 代码满足以下条件，则它是封闭的：

1. 不含 `import` 语句
2. `__builtins__` 为空
3. 所有属性读取（`.attr`）经过白名单检查，白名单由 TypeSpec 声明或 Declaration 声明定义
4. 白名单中声明的每个属性，其返回值类型不会引入新的逃逸路径（即返回的对象要么是基础类型，要么是宿主注入的已知类型）。注：此条件是充分条件而非必要条件——即使某个方法返回了未注册类型的对象，守卫函数的 deny-by-default 行为也会阻断后续属性访问（见下文补充）
5. 白名单检查函数本身不可被代码修改

**证明思路**：

代码能接触到的对象的来源只有三种：
- (a) 宿主通过 globals 注入的符号
- (b) 代码自身创建的对象（字面量、函数定义、局部变量）
- (c) 从 (a)(b) 的对象通过操作派生的对象

对于 (c)，派生操作只有两类：
- 非属性操作（运算符、下标、迭代等）：返回类型可预测，不引入新的未知对象
- 属性访问：经过白名单检查，只返回声明范围内的属性值

由条件 4，白名单中的属性值本身也是已知类型的对象，递归满足同样的约束。

**补充**：即使出现未注册类型的对象（某个白名单方法返回了没有 TypeSpec 的类型），守卫函数的默认行为是**拒绝**（deny-by-default）——`type_whitelist.get(obj_type)` 返回 `None`，属性访问被阻断。因此类型系统的封闭性不依赖"所有返回类型都有 TypeSpec"，而是依赖守卫函数的默认拒绝行为。TypeSpec 的完整性影响的是**可用性**（agent 能做什么），不影响**安全性**（agent 不能做什么）。

由条件 1 和 2，不存在其他引入新对象的途径。

由条件 5，白名单检查不可被绕过。

∴ 代码不可能获取到声明范围之外的对象引用。∎

## 实现机制

### 子集语言定义

#### AST 语法限制

以下语法在 AST 层面被禁止（遇到对应节点直接拒绝）：

| 禁止语法 | AST 节点 | 原因 |
|----------|---------|------|
| `import` / `from ... import` | `Import`、`ImportFrom` | 引入新能力，破坏封闭性 |

所有其他 Python 语句和表达式均允许：

| 语句 | 说明 |
|------|------|
| 赋值 | `x = expr`、`x += expr`、解包赋值 `a, b = expr`、walrus `:=` |
| 控制流 | `if`/`elif`/`else`、`for`/`while`/`break`/`continue` |
| 函数定义 | `def`（局部函数，含嵌套）、`lambda`、`async def` |
| 类定义 | `class`（安全性由 `_safe_build_class` 保证，见受限内置函数） |
| 返回 | `return`、`yield`、`yield from` |
| 异常 | `raise`/`try`/`except`/`finally` |
| 上下文管理 | `with`/`async with` |
| 模式匹配 | `match`/`case` |
| 作用域 | `global`、`nonlocal` |
| 异步 | `await`、`async for` |
| 其他 | `assert`、`pass`、`del` |

**所有表达式均允许**：字面量、运算符、下标、函数调用、属性访问、推导式、生成器表达式、f-string、星号解包等。

安全性不依赖语法限制（import 除外），而依赖属性访问白名单。`class` 语句的安全性由 `_safe_build_class` 保证（见受限内置函数）。

#### AST 转换规则

属性读取和赋值均经过白名单检查：

```python
# 属性读取：obj.attr → getattr(obj, 'attr')
Attribute(value, attr, Load) → Call(Name('getattr'), [value, Constant(attr)])

# 属性赋值：obj.attr = val → setattr(obj, 'attr', val)
Attribute(value, attr, Store) → Call(Name('setattr'), [value, Constant(attr), val])

# 属性删除：del obj.attr → 不转换，原样执行
```

对称守卫使心智模型更简单：**白名单定义了类型上可以碰的属性，读写统一**。属性删除不守卫——删除是罕见操作，且不产生引用。

#### 守卫函数

```python
def _make_guards(type_whitelist, safe_overrides):
    def _getattr_guard(obj, name):
        """属性读取守卫"""
        obj_type = type(obj)
        allowed = type_whitelist.get(obj_type)
        if allowed is None or name not in allowed:
            raise AttributeError(
                f"'{obj_type.__name__}' object has no attribute '{name}'"
            )
        # 危险方法 → 透明替换为安全版本
        override = safe_overrides.get((obj_type, name))
        if override is not None:
            return override.__get__(obj, obj_type)
        return getattr(obj, name)

    def _setattr_guard(obj, name, value):
        """属性赋值守卫"""
        obj_type = type(obj)
        allowed = type_whitelist.get(obj_type)
        if allowed is None or name not in allowed:
            raise AttributeError(
                f"'{obj_type.__name__}' object has no attribute '{name}'"
            )
        setattr(obj, name, value)

    return _getattr_guard, _setattr_guard
```

注：`_getattr_guard` 内部调用原生 `getattr` 获取属性值——这发生在守卫函数内部（已通过白名单检查），与 globals 中的 `getattr`（指向守卫自身）不冲突。守卫函数在构造时通过闭包捕获原生 `getattr` 引用。

#### 受限内置函数

部分内置函数存在多态调用风险，需包装为安全版本：

| 函数 | 风险 | 安全版本 |
|------|------|---------|
| `type` | 三参数调用 `type('X', (object,), {...})` 绕过 `_safe_build_class` 的基类检查 | 只允许单参数调用（类型查询），其余抛出 `TypeError` |
| `__build_class__` | `class` 语句的底层机制，需验证基类合法性并注册新类型白名单 | 验证基类在白名单中，创建后自动注册新类型 |

```python
_builtin_type = type

def _safe_type(*args):
    """只允许 type(obj) 类型查询，禁止 type(name, bases, dict) 动态建类"""
    if len(args) != 1:
        raise TypeError("type() takes 1 argument")
    return _builtin_type(args[0])
```

#### `_safe_build_class`

`class` 语句编译为 `LOAD_BUILD_CLASS` 字节码，从 `__builtins__` 中查找 `__build_class__`。沙箱将安全版本放入 `__builtins__`：

```python
{'__builtins__': {'__build_class__': _safe_build_class}}
```

`_safe_build_class` 与守卫函数共享 `type_whitelist` 引用（通过闭包）：

```python
from builtins import __build_class__ as _real_build_class

def _safe_build_class(func, name, *bases, **kwds):
    """安全的 __build_class__：验证基类，注册新类型白名单。"""
    if 'metaclass' in kwds and kwds['metaclass'] is not type:
        raise TypeError("custom metaclass is not supported")
    for base in bases:
        if base is not object and base not in type_whitelist:
            raise TypeError(f"cannot subclass '{base.__name__}'")
    cls = _real_build_class(func, name, *bases, **kwds)
    # 新类型白名单 = 父类白名单并集 + 类体中定义的属性名
    allowed = set()
    for base in bases:
        if base in type_whitelist:
            allowed |= type_whitelist[base]
    allowed |= set(cls.__dict__)
    type_whitelist[cls] = frozenset(allowed)
    return cls
```

**安全论证**：

1. **基类验证**：所有基类必须在 `type_whitelist` 中（或为 `object`）。agent 无法继承未知类型
2. **白名单继承**：新类型的白名单 = 父类白名单 ∪ 类体定义。agent 无法通过子类访问父类 TypeSpec 之外的属性
3. **类体安全**：类体代码经过 AST 转换，属性访问受守卫约束。方法的 `__globals__` 指向 ScriptEnv
4. **C 层面操作**：`__build_class__` 和 `type.__new__` 内部的 C 层面属性访问（确定 metaclass、调用 `__init_subclass__`、`__set_name__`）不暴露中间产物给 agent
5. **`__builtins__` 原地修改不构成逃逸**：agent 可替换 `__builtins__['__build_class__']`，但替换物只能是 agent 已有对象，不产生新能力。且替换物无法访问 `_real_build_class`（闭包内）和 `type_whitelist`（闭包内）

**纯沙箱类型的属性访问**：守卫函数对 `type_whitelist` 中注册的类型按白名单放行。由于新类型的白名单包含类体所有定义，agent 可正常使用自定义类的方法和类属性。实例属性（如 `__init__` 中 `self.x = 1`）通过 `setattr` 守卫检查同一白名单——`__init__` 在类体中定义，因此在白名单中；但 `x` 不在类体 `__dict__` 中（运行时赋值），会被 `setattr` 守卫拒绝。

**实例属性的解决方案**：对纯沙箱类型（所有基类均为 `object` 或其他沙箱类型），可放宽为不限制属性访问——实例只包含 agent 已有对象，不引入新能力。对继承 host 类型的沙箱类型，仍需严格白名单。具体实现策略留待实施时确定。

### 条件 5 的实现：守卫函数不可覆盖

白名单检查函数通过 AST 转换注入——所有 `obj.attr` 读取被转换为 `getattr(obj, 'attr')` 调用。`getattr` 在 globals 中指向守卫函数，由 ScriptEnv 冻结保护。

**设计选择**：使用合法标识符 `getattr` 而非非法标识符 `.getattr`。

非法标识符方案（`.getattr`）利用 Python 源码无法表达以 `.` 开头的名称来阻止覆盖——源码层面不可引用，条件 5 天然满足，冻结仅为纵深防御。

合法标识符方案（`getattr`）将守卫函数同时作为 agent 可用的 builtin——`obj.attr` 和 `getattr(obj, 'name')` 调用同一个守卫。条件 5 的满足依赖 ScriptEnv 的冻结机制（见条件 2 的封堵方案），冻结从纵深防御变为必要条件。

两个方案安全性等价——条件 5 均满足。合法标识符方案的优势：

- agent 可直接调用 `getattr(obj, 'name')` 探索环境（动态属性名）
- 消除非法标识符的认知开销和工具兼容性风险
- ScriptEnv 冻结机制已经过 `__builtins__` 场景验证，复用同一机制

**覆盖尝试分析**：

- `getattr = evil` → ScriptEnv.\_\_setitem\_\_ 拦截冻结键，静默忽略
- `del getattr` → ScriptEnv.\_\_delitem\_\_ 拦截，静默忽略
- `globals()['getattr'] = evil` → 同上，ScriptEnv 拦截
- `dict.__setitem__(globals(), 'getattr', evil)` → `dict.__setitem__` 是属性访问 → `getattr(dict, '__setitem__')` → 守卫检查白名单 → `type` 类型无 `__setitem__` 声明 → AttributeError

**不变量**：`type` 类型的 TypeSpec 白名单**绝对不能**包含 `__setitem__`、`__delitem__`、`__getitem__` 等 dict 操作方法。否则 `dict.__setitem__(globals(), 'getattr', evil)` 可以绕过 ScriptEnv 冻结，整个安全模型崩塌。这是 ScriptEnv 封闭性的前提条件。

### 条件 2 的实现：`__builtins__` 回填漏洞与封堵

条件 2 要求 `__builtins__` 为空。实施方式是在 globals 中放入 `{'__builtins__': {}}`。但 CPython 的 `exec` 存在一个隐式行为：**如果 globals 字典中不存在 `__builtins__` 键，CPython 会自动注入当前解释器的真实 `builtins` 模块。**

这意味着如果沙箱代码执行 `del __builtins__`（从 globals 字典中删除该键），后续的 `exec` 调用（REPL 模式的下一步）会触发 CPython 回填，使 `eval`、`exec`、`__import__` 等危险函数全部可用。

**攻击路径**：

```
# REPL step 1
del __builtins__           # globals 中 __builtins__ 键被删除

# REPL step 2（新的 exec 调用，同一个 globals）
# CPython 发现 globals 中没有 __builtins__ → 注入真实 builtins
eval("__import__('os').system('...')")   # 逃逸成功
```

**为什么单步 exec 内不受影响**：CPython 在创建 frame 时缓存 builtins 引用。同一 frame 内修改 `globals['__builtins__']` 不影响当前 frame 的名称查找。威胁仅存在于 REPL 模式的跨步骤场景。

**封堵方案：ScriptEnv**

使用自定义 dict 子类作为 globals 字典，在字典层面冻结关键键的删除和覆盖：

```python
class ScriptEnv(dict):
    """沙箱 globals 字典，冻结安全关键键的删除和覆盖。"""
    _FROZEN = frozenset({'__builtins__', 'getattr'})

    def __delitem__(self, key):
        if key in self._FROZEN:
            return  # 静默忽略
        super().__delitem__(key)

    def __setitem__(self, key, value):
        if key in self._FROZEN:
            return  # 静默忽略
        super().__setitem__(key, value)
```

**`setattr` 是否需要冻结**：属性赋值不构成逃逸（见下文"属性赋值与删除的安全性分析"）。`setattr` 守卫的目的是行为一致性，不是安全性。即使 agent 覆盖了 `setattr`，也只是跳过写入白名单检查，不违反封闭性。因此 `setattr` 不需要冻结——冻结集合仅包含 `__builtins__` 和 `getattr`。

**为什么在字典层而非 AST 层封堵**：对 `__builtins__` 的写操作在 AST 层有无穷多种表达——`del __builtins__`、`__builtins__ = x`、`for __builtins__ in ...`、`except E as __builtins__`、`(__builtins__ := x)` 等等。每种都是一个特殊规则。而所有这些语法最终都经过 `dict.__setitem__` 或 `dict.__delitem__`，在字典层一次性封堵。

**`__builtins__` 原地修改不构成逃逸**：agent 代码可以执行 `__builtins__["eval"] = 42`，往空字典里塞值。但 agent 只能塞它已经能引用到的对象（如 `len`、`42`、`"hello"`），而这些对象中没有危险函数。原地修改不产生新能力，不构成权限逃逸，因此不需要防范。

### 属性赋值与删除的安全性分析

属性赋值（`obj.attr = val`）和属性删除（`del obj.attr`）改变对象状态，但不会让代码获得新的对象引用。逃逸的本质是"读取到不安全对象"，不是"修改对象"。因此从安全角度，只需守卫属性**读取**。

**注意**：虽然赋值不构成逃逸，但实施层面仍可能选择对赋值施加白名单检查——原因不是安全性，而是行为一致性（允许写入却禁止读取同一属性会造成困惑）。这属于实施决策，不影响安全证明。

### 绕过 AST 的属性访问

某些白名单方法内部包含独立的属性访问语言，在 C 运行时层面执行，绕过 AST 转换。这类方法需要在白名单检查时透明替换为安全版本。

**当前已知的绕过路径**：仅 `str.format` / `str.format_map` 的 format 迷你语言（支持 `{0.__class__}` 等属性遍历语法）。其他可能包含类似机制的模块（如 `re`、`pickle`、`ctypes`）不在子集的注入范围内，因此不构成威胁。f-string 的转换标志（`!s`/`!r`/`!a`）调用 `__str__`/`__repr__`/`__ascii__`，这些是 dunder 方法，由 Python 运行时直接调度，不经过 `.attr` 语法，因此不构成绕过。f-string 中的属性访问（如 `f"{obj.name}"`）在 AST 层可见，会被正常转换为守卫调用。

**透明替换方案**：守卫函数在返回属性前查找替换表，命中则返回安全版本的绑定方法：

```python
import string

class SafeFormatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
        # get_field 是 format 迷你语言中属性/下标遍历的唯一入口
        if '.' in field_name or '[' in field_name:
            raise AttributeError(
                f"'{field_name}' is not a valid format field"
            )
        return super().get_field(field_name, args, kwargs)

_safe_fmt = SafeFormatter()

# 类型 → 方法名 → 安全实现
_safe_overrides = {
    (str, 'format'): lambda self, *a, **kw: _safe_fmt.format(self, *a, **kw),
    (str, 'format_map'): lambda self, mapping: _safe_fmt.vformat(self, (), mapping),
}
```

用户代码写法不变：
- `"hello {}".format(name)` — 正常工作
- `"{:.2f}".format(3.14)` — 正常工作
- `"{0.__class__}".format(obj)` — `AttributeError`

这不影响封闭性定理的成立——定理的条件 4 要求"白名单中的属性不引入逃逸路径"，透明替换确保了这一条件对所有声明的属性成立。

### 异常策略

沙箱**不使用自定义异常类**，所有错误均抛出标准 Python 异常：

| 场景 | 异常类型 | 示例消息 |
|------|---------|---------|
| 属性访问被拒 | `AttributeError` | `'tuple' object has no attribute '__class__'` |
| AST 禁止语法 | `SyntaxError` | `import statements are not supported` |
| `type()` 多参数调用 | `TypeError` | `type() takes 1 argument` |
| 未注入的函数 | `NameError` | `name 'eval' is not defined`（CPython 原生行为，无需代码） |

**设计原因**：沙箱的使用者是 agent（LLM）。标准异常让 agent 认为"这个东西不存在"或"用法不对"，从而转向可用接口。自定义异常（如 `SubsetViolation`）会让 agent 意识到"存在但被限制"，可能浪费 token 尝试绕过。消息风格与 CPython 原生错误一致，进一步强化"不存在"的认知。

### REPL 模式

通过共享 `locals` dict 实现跨调用状态保持，无需专门的 Session 类：

- 不传 `locals` 时，`locals=globals`（和 Python `exec` 一致），变量存入 globals
- 传入同一个 `locals` dict = REPL 会话（跨步骤共享变量）
- 不同的 `locals` dict = 独立会话（共享同一套能力但状态隔离）
- 会话生命周期由宿主管理（丢弃 dict 即销毁会话）

## Builtin 函数安全分析

基于封闭性定理，对 Python 所有 builtin 函数逐一分析是否存在逃逸路线。**方法论**：对每个 builtin，尝试构造一条违反封闭性的执行路径（获取声明范围之外的对象引用）。能构造出逃逸路径的必须排除；构造不出的安全可用。

**分析前提**：封闭性定理的 5 个条件全部满足——AST 禁止 import、`__builtins__` 受 ScriptEnv 冻结保护、所有属性读取经过 `getattr` 守卫、白名单由 TypeSpec 声明、守卫函数不可覆盖、`class` 由 `_safe_build_class` 保护。

### 必须排除的 builtin（存在逃逸路线）

**`eval` / `exec`**

逃逸路径：编译并执行代码字符串，不经过 AST 转换。编译后的字节码使用原生 `LOAD_ATTR` 指令，属性访问不经过 `getattr` 守卫。

```python
eval("obj.__class__.__bases__[0].__subclasses__()")
# .__class__ → LOAD_ATTR，绕过守卫 → 逃逸
```

违反条件 3（所有属性读取经过白名单检查）。

**`compile`**

`compile` 产生未经 AST 转换的 code object。单独存在时无法执行（Python 中执行 code object 的唯一途径是 `exec`/`eval`，`types.FunctionType` 不可 import）。但因其唯一用途是支持代码执行，无其他合法用途，排除。

**`__import__`**

逃逸路径：直接加载外部模块，引入沙箱外的任意对象。

```python
os = __import__('os')  # 获得 os 模块 → 完全逃逸
```

违反封闭性定义（引入非注入、非派生的对象）。

**`vars(obj)`**

逃逸路径：原生 `vars(obj)` 在 C 层面调用 `PyObject_GenericGetDict(obj)`，直接返回 `obj.__dict__`，不经过 `getattr` 守卫。返回的 dict 包含未在白名单中声明的实例属性。

```python
d = vars(injected_obj)   # 绕过守卫获取 __dict__
d['_internal_ref']        # 访问未声明的属性值
```

对比：源码中 `obj.__dict__` 经过 AST 转换为 `getattr(obj, '__dict__')` → 守卫检查白名单 → 若 `__dict__` 未声明则 AttributeError。`vars(obj)` 在 C 层面绕过了这一检查。

违反条件 3。

**安全版本**：将 `vars(obj)` 路由到守卫，`__dict__` 的可访问性由 TypeSpec 白名单控制：

```python
def _safe_vars(*args):
    if len(args) == 0:
        raise TypeError("vars() 不带参数请用 locals()")
    if len(args) == 1:
        return getattr(args[0], '__dict__')  # 走守卫
    raise TypeError(f"vars expected at most 1 argument, got {len(args)}")
```

无参 `vars()` 不支持——语义等价于 `locals()`，但在包装函数内部调用 `locals()` 返回的是 `_safe_vars` 的局部变量，非调用者作用域。agent 应直接使用 `locals()`。

### 非逃逸但需排除的 builtin

以下 builtin 不违反封闭性，但因副作用或运行安全需排除：

| builtin | 原因 |
|---------|------|
| `open` | 文件系统副作用。即使返回的 file 对象因无 TypeSpec 而无法调用方法，`open(path, 'w')` 仍会截断文件 |
| `breakpoint` | 通过 C 层面调用 `sys.breakpointhook()`，交互式环境中调试器具有不受限的 eval 能力 |
| `input` | 等待 stdin 输入，非交互环境中永久阻塞 |
| `exit` / `quit` | 终止解释器进程 |

### 安全 builtin 的证明

以下证明其余 Python builtin 不存在违反封闭性的执行路径。

**getattr（守卫版本）**

`getattr` 在 globals 中指向守卫函数，由 ScriptEnv 冻结。agent 调用 `getattr(obj, 'name')` 与源码中 `obj.name` 等价——两者最终调用同一个守卫函数，受白名单约束。覆盖尝试均被 ScriptEnv 拦截（见条件 5 的实现）。不存在逃逸路线。

**setattr / delattr**

属性赋值（`setattr`）和删除（`delattr`）改变对象状态，不产生新的对象引用。逃逸的本质是"读取到不安全对象"。`setattr(obj, name, value)` 中 `value` 来自 agent 已有对象，不引入新能力。`delattr(obj, name)` 删除属性，不产生引用。

即使提供原生版本（绕过写入守卫），也不违反封闭性——封闭性仅约束属性**读取**。

**hasattr**

`hasattr(obj, name)` 在 C 层面调用 `PyObject_GetAttr`，绕过守卫。但返回值是 `bool`——agent 获得属性存在性信息，但无法获取属性值本身。属性存在性是信息泄露（与 `dir` 类似），不构成逃逸（无对象引用泄露）。

**dir**

`dir(obj)` 返回属性名字符串列表。暴露对象的全部属性名，但 agent 仍无法绕过守卫访问非白名单属性。信息泄露不构成逃逸。

**globals / locals**

`globals()` 返回 ScriptEnv 引用。agent 可见守卫函数和注入的命名空间，但冻结键不可修改。尝试通过底层 dict 方法绕过冻结：

```python
dict.__setitem__(globals(), 'getattr', evil)
# → getattr(dict, '__setitem__') → 守卫检查 type 的白名单
# → __setitem__ 不在 type 白名单 → AttributeError
```

底层 dict 方法的访问本身经过守卫，形成封闭。不存在逃逸路线。

`locals()` 返回局部变量 dict，不含安全敏感项。不存在逃逸路线。

**type（单参数限制）**

`_safe_type` 仅允许 `type(obj)` 查询类型，返回类型对象。类型对象的属性访问经过守卫（`type(obj).__bases__` → `getattr(type_obj, '__bases__')` → 白名单检查）。三参数调用被 `_safe_type` 拒绝（`TypeError`）。不存在逃逸路线。

**object**

`object()` 创建空实例。`object.__getattribute__` 等属性访问经过守卫 → `type(object)` = `type` → `__getattribute__` 不在 `type` 白名单 → AttributeError。无法用于构造不受限的属性访问。

**类型构造器**

`int`, `float`, `str`, `bool`, `list`, `dict`, `tuple`, `set`, `frozenset`, `bytes`, `bytearray`, `complex`, `memoryview`, `slice`

创建对应类型的实例。实例的属性访问经过守卫，受 TypeSpec 白名单约束。构造过程不引入外部对象。

**纯函数**

`len`, `range`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `reversed`, `min`, `max`, `sum`, `any`, `all`, `abs`, `round`, `repr`, `print`, `hash`, `id`, `iter`, `next`, `callable`, `chr`, `ord`, `hex`, `bin`, `oct`, `pow`, `divmod`, `format`, `ascii`, `isinstance`, `issubclass`, `aiter`, `anext`

接受沙箱内对象，返回基础类型值或可预测类型的对象。这些函数在 C 层面调用 dunder 方法（`__len__`、`__iter__` 等），不经过 `.attr` 语法，由 Python 运行时直接调度。返回值的后续属性访问仍受守卫约束。不引入外部对象。

**异常类**

`Exception`, `ValueError`, `TypeError`, `NameError`, `AssertionError`, `OSError`, `FileNotFoundError`, `PermissionError`, `UnicodeError`, `UnicodeDecodeError`, `UnicodeEncodeError`, `ArithmeticError`, `LookupError`, `EOFError`, `StopIteration`, `StopAsyncIteration`, `GeneratorExit`, `RecursionError`, `BufferError`, `SystemError`, `NotImplementedError`, `OverflowError`, `ZeroDivisionError`, `IndexError`, `KeyError`, `AttributeError`, `RuntimeError`, `FloatingPointError`, `BlockingIOError`, `ConnectionError`, `ConnectionAbortedError`, `ConnectionRefusedError`, `ConnectionResetError`, `TimeoutError`, `BrokenPipeError`, `ChildProcessError`, `IsADirectoryError`, `NotADirectoryError`, `ProcessLookupError`, `InterruptedError`, `Warning`, `DeprecationWarning`, `UserWarning`, `SyntaxWarning`, `RuntimeWarning`, `FutureWarning`, `ImportWarning`, `UnicodeWarning`, `ResourceWarning`, `BytesWarning` 等

异常类是类型。实例化产生异常对象（数据容器，`.args` 等属性受守卫约束）。不引入外部对象，不执行不受限的属性访问。

**描述符相关**

`property`, `classmethod`, `staticmethod`, `super`

`class` 语句允许后，这些函数在类体内可正常使用。`property` 创建描述符，`classmethod`/`staticmethod` 包装方法——它们的底层函数均在沙箱内定义（AST 转换、ScriptEnv globals），不引入新能力。

`super()` 零参数形式依赖编译器隐式创建的 `__class__` cell，在类方法内可用。`super()` 返回 super proxy，其属性访问经守卫检查。由于子类白名单继承了父类白名单，`super().method()` 对白名单内方法正常工作。不构成逃逸。

### 安全分析总结

| 分类 | 结论 | 判定依据 |
|------|------|---------|
| `eval`, `exec`, `compile` | 排除 | 绕过 AST 转换，属性访问不经过守卫 |
| `__import__` | 排除 | 引入外部模块，违反封闭性 |
| `vars` | 排除（原生）/ **安全**（包装版） | 原生绕过守卫；包装版路由到 `getattr(obj, '__dict__')` |
| `open`, `breakpoint`, `input`, `exit`, `quit` | 排除 | 非逃逸，但有文件系统副作用 / 调试器注入 / 阻塞 / 终止 |
| `__build_class__`（安全版本） | **安全** | 验证基类在白名单中，自动注册新类型白名单 |
| `getattr`（守卫版本） | **安全** | 守卫函数本身，由 ScriptEnv 冻结 |
| `setattr`, `delattr` | **安全** | 属性写入/删除不产生对象引用 |
| `hasattr`, `dir` | **安全** | 信息泄露（bool / 字符串列表），无对象引用泄露 |
| `globals`, `locals` | **安全** | ScriptEnv 冻结保护，底层 dict 方法访问经过守卫 |
| 其余所有 builtin | **安全** | 返回基础类型值，后续属性访问受守卫约束 |

## 测试计划

### 正向测试：合法代码正常执行

- 基础类型操作：`"hello".upper()`、`[1,2,3].append(4)`、`{"a":1}.get("b", 0)`
- 控制流：循环、条件、异常处理、函数定义
- 宿主注入接口调用
- REPL 跨步骤状态保持（共享 locals dict）
- f-string 内的属性访问（经过 AST 转换）
- `type(obj)` 单参数类型查询

### 逃逸测试：已知攻击路径全部被拦截

所有逃逸尝试应产生标准 Python 异常，消息风格与 CPython 一致：

- dunder 链逃逸：`().__class__.__bases__[0].__subclasses__()` → `AttributeError: 'tuple' object has no attribute '__class__'`
- frame 逃逸：`(lambda: 0).__code__` → `AttributeError: 'function' object has no attribute '__code__'`
- import 语句：`import os` → `SyntaxError`
- 动态建类：`type('X', (object,), {})` → `TypeError: type() takes 1 argument`
- 非法基类：`class Evil(non_whitelisted_type): pass` → `TypeError: cannot subclass 'X'`
- 自定义 metaclass：`class Evil(metaclass=MyMeta): pass` → `TypeError: custom metaclass is not supported`
- format 逃逸：`"{0.__class__}".format(obj)` → `AttributeError`
- 守卫覆盖尝试：确认 `getattr` 不可被覆盖或删除

### `__builtins__` 攻击路径测试

验证 ScriptEnv 正确封堵所有 `__builtins__` 相关攻击：

- `del __builtins__` → 静默忽略，`__builtins__` 仍存在于 globals 中
- `__builtins__ = {"eval": 42}` → 静默忽略，`__builtins__` 不变
- `for __builtins__ in [1,2,3]: pass` → 静默忽略
- `except E as __builtins__` → 静默忽略
- `(__builtins__ := {"eval": 42})` → 静默忽略
- `__builtins__["eval"] = len` → 允许（原地修改不构成逃逸，agent 只能塞已有对象）
- REPL 跨步骤验证：step1 执行 `del __builtins__`，step2 确认危险函数仍不可用
- `getattr` 键同样不可删除/覆盖

### 对称性测试：读写行为一致

- 白名单内属性：读写均成功
- 白名单外属性：读写均拒绝，错误信息一致（`AttributeError`）
- 未注册类型：读写均拒绝（`AttributeError`）

### 边界测试

- 嵌套属性访问：`obj.a.b.c` 每一步都经过守卫
- 属性删除：不经过守卫，正常执行
- 描述符交互：白名单属性是 property 时的行为

### class 安全测试

- 纯沙箱类：`class Foo: ...` → 创建成功，实例方法可调用
- 合法继承：`class Child(WhitelistedParent): ...` → 创建成功
- 白名单继承验证：子类可访问父类白名单内属性，不可访问白名单外属性
- 类体方法在沙箱内执行：方法内的属性访问经过守卫
- `__build_class__` 替换不构成逃逸：`__builtins__['__build_class__'] = agent_func` → class 行为改变但无逃逸
- `super()` 在方法中正常工作：`super().method()` 对白名单内方法可用

### 性能基线测试

AST 转换后每次属性访问多一次函数调用 + 字典查找。建立基线认知，不设硬指标：

- 循环密集场景：`for i in range(10000): "hello".upper()` — 对比转换前后耗时
- 嵌套属性场景：`obj.a.b.c` 三次守卫调用的累积开销
- 纯计算场景（无属性访问）：`sum(range(10000))` — 确认无额外开销

## 已知不处理的边界

以下场景在当前设计中**不处理**，记录于此供未来评估：

| 场景 | 分析 | 影响 |
|------|------|------|
| `__del__` 回调 | agent 可在自定义类中定义 `__del__`，GC 回收时触发。方法的 `__globals__` 指向 ScriptEnv，代码经 AST 转换 | 不构成逃逸——`__del__` 方法在沙箱环境中执行，属性访问受守卫约束。GC 时机不可预测，但不影响安全性 |
| `@impl` 返回类型安全 | Declaration 子类的 `@impl` 可能返回任意类型的对象 | `@impl` 作者对返回值安全性负责，不在沙箱保证范围内。守卫的 deny-by-default 提供了保底：即使返回未知类型，其属性访问仍被拒绝 |
| 资源限制（CPU/内存） | `while True: pass` 或 `range(10**18)` 等代码会耗尽 CPU/内存 | 不属于逃逸问题，由宿主通过超时、内存上限等机制处理。沙箱保证的是"不能做什么"（能力边界），不保证"不会消耗多少资源" |
| 生成器/协程 frame 对象 | `gen.gi_frame`、`gen.gi_code`、`coro.cr_frame` 等属性可暴露 frame 和 code 对象，进而访问局部变量和字节码 | 不构成逃逸——这些属性的访问经过守卫，`generator`/`coroutine` 类型默认无 TypeSpec，deny-by-default 阻断。为这些类型添加 TypeSpec 时**不能**将 `gi_frame`、`gi_code`、`cr_frame`、`cr_code` 加入白名单 |
| `memoryview` 数据篡改 | `memoryview` 的切片/下标操作（不经过守卫）可直接读写底层 buffer，如果宿主注入了 `bytearray` 等支持 buffer protocol 的对象，agent 可绕过对象接口直接修改内存 | 不构成逃逸（不获取新对象引用），但可能造成数据篡改。沙箱的安全目标是证明不存在逃逸路径，数据完整性由宿主负责（不注入可写 buffer 对象，或不将 `memoryview` 加入白名单 builtin） |
