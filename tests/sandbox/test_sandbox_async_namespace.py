"""NamespaceTools async wrapper 线程/loop 调度回归测试。

覆盖三条调用路径，对应 bugfix-namespace-tools-async-main-loop.md：

1. ``_wrap_async`` 同步路径（pysandbox tool entry）—— worker 线程同步
   调用，coroutine 必须回到主 loop 执行；未 ``bind_main_loop()`` 抛错；
   目标 loop 同线程同步调用抛错。
2. ``_mcp_share.py:_handle_call`` 异步路径（pysandbox/namespaces.call RPC）——
   handler 已在主 loop async 上下文，应通过 ``_async_original`` 直接
   await 原 coroutine，绕过 sync wrapper，**不依赖 bind_main_loop()**。
3. ``SandboxEnv.bind_main_loop()`` helper —— 写入 ``_async_loop`` /
   ``_async_loop_thread_id``，幂等。
"""

import asyncio
import threading

import pytest

from mutio.mcp.protocol import JsonRpcDispatcher
from mutagent.sandbox import SandboxEnv
from mutagent.sandbox._env_impl import (
    _wrap_async,
    sandbox_env_add_namespace,
)
from mutagent.sandbox._namespace_impl import Namespace
from mutagent.sandbox._mcp_share import register_pysandbox_methods


# ---------------------------------------------------------------------------
# 1. _wrap_async 同步路径 —— pysandbox tool entry
# ---------------------------------------------------------------------------

class TestWrapAsyncSyncPath:

    @pytest.mark.asyncio
    async def test_wrapper_routes_to_main_loop_when_called_from_worker(self):
        """worker 线程同步调用 wrapper 时，coroutine 必须在主 loop 线程执行。"""
        sandbox = SandboxEnv()
        sandbox.bind_main_loop()

        async def probe():
            return {
                "thread_id": threading.get_ident(),
                "loop": asyncio.get_running_loop(),
            }

        wrapped = _wrap_async(sandbox, probe)
        main_loop = asyncio.get_running_loop()
        main_thread_id = threading.get_ident()
        result_box: dict = {}

        def worker():
            result_box["worker_thread_id"] = threading.get_ident()
            result_box["result"] = wrapped()

        await main_loop.run_in_executor(None, worker)

        assert result_box["worker_thread_id"] != main_thread_id
        assert result_box["result"]["thread_id"] == main_thread_id
        assert result_box["result"]["loop"] is main_loop

    def test_wrapper_requires_bind_main_loop(self):
        """未 bind 时不再 fallback 到 asyncio.run，明确报生命周期错误。"""
        sandbox = SandboxEnv()

        async def hello():
            return "ok"

        wrapped = _wrap_async(sandbox, hello)

        with pytest.raises(RuntimeError, match="_async_loop not set"):
            wrapped()

    @pytest.mark.asyncio
    async def test_wrapper_rejects_sync_call_from_target_loop_thread(self):
        """目标 loop 线程内同步调用 wrapper 会死锁，因此应提前报错。"""
        sandbox = SandboxEnv()
        sandbox.bind_main_loop()

        async def hello():
            return "ok"

        wrapped = _wrap_async(sandbox, hello)

        with pytest.raises(RuntimeError, match="Cannot synchronously call async"):
            wrapped()

    def test_wrapper_exposes_async_original(self):
        """wrapper 必须挂 _async_original，供异步上下文绕过 sync wrapper。"""
        sandbox = SandboxEnv()

        async def hello(**kwargs):
            return ("ok", kwargs)

        wrapped = _wrap_async(sandbox, hello)
        assert getattr(wrapped, "_async_original", None) is hello


# ---------------------------------------------------------------------------
# 2. _mcp_share.py:_handle_call 异步路径 —— pysandbox/namespaces.call RPC
# ---------------------------------------------------------------------------

class TestHandleCallAsyncPath:

    @pytest.mark.asyncio
    async def test_handle_call_awaits_async_original_without_bind(self):
        """_mcp_share.py 路径已在主 loop async 上下文，应直接 await 原 coroutine，
        即使 SandboxEnv 从未 bind_main_loop() 也必须工作（不能踩 sync wrapper）。

        这是导致用户看到 ``MCP error -32603: SandboxEnv._async_loop not set``
        的根因路径，必须有回归覆盖。
        """
        sandbox = SandboxEnv()
        # 故意不 bind_main_loop()：验证 _mcp_share.py 路径不依赖它

        main_loop = asyncio.get_running_loop()
        main_thread_id = threading.get_ident()

        async def probe(**kwargs):
            return {
                "thread_id": threading.get_ident(),
                "kwargs": kwargs,
            }

        # 用 _wrap_async 模拟 _build_declaration_namespaces 的产物
        wrapped = _wrap_async(sandbox, probe)
        ns = Namespace("probe", description="probe ns")
        ns.register("where", wrapped, "")
        sandbox_env_add_namespace(sandbox, ns)

        dispatcher = JsonRpcDispatcher()
        register_pysandbox_methods(dispatcher, sandbox)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "pysandbox/namespaces.call",
            "params": {
                "namespace": "probe",
                "name": "where",
                "arguments": {"foo": "bar"},
            },
        }
        response = await dispatcher.handle(request)

        assert "error" not in response, response
        result = response["result"]
        # 在主 loop async handler 里 await，所以观察到的是主 loop 线程
        assert result["thread_id"] == main_thread_id
        assert result["kwargs"] == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_handle_call_falls_back_to_sync_fn(self):
        """非 async-wrapper 的普通 sync 函数仍按 fn(**arguments) 调用。"""
        sandbox = SandboxEnv()

        def add(**kwargs) -> int:
            return kwargs["a"] + kwargs["b"]

        ns = Namespace("math", description="")
        ns.register("add", add, "")
        sandbox_env_add_namespace(sandbox, ns)

        dispatcher = JsonRpcDispatcher()
        register_pysandbox_methods(dispatcher, sandbox)

        response = await dispatcher.handle({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "pysandbox/namespaces.call",
            "params": {
                "namespace": "math",
                "name": "add",
                "arguments": {"a": 3, "b": 4},
            },
        })
        assert response["result"] == 7


# ---------------------------------------------------------------------------
# 3. bind_main_loop helper
# ---------------------------------------------------------------------------

class TestBindMainLoop:

    @pytest.mark.asyncio
    async def test_bind_main_loop_writes_loop_and_thread_id(self):
        sandbox = SandboxEnv()
        sandbox.bind_main_loop()

        assert getattr(sandbox, "_async_loop") is asyncio.get_running_loop()
        assert getattr(sandbox, "_async_loop_thread_id") == threading.get_ident()

    @pytest.mark.asyncio
    async def test_bind_main_loop_is_idempotent(self):
        sandbox = SandboxEnv()
        sandbox.bind_main_loop()
        loop1 = sandbox._async_loop  # type: ignore[attr-defined]
        sandbox.bind_main_loop()
        loop2 = sandbox._async_loop  # type: ignore[attr-defined]
        assert loop1 is loop2
