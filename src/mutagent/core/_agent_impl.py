"""mutagent.core._agent_impl -- Agent main loop implementation."""

import asyncio
import inspect
import logging
import time
from contextvars import ContextVar
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

import mutobj
from .agent import Agent, CancelFn
from .messages import (
    Message,
    Response,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    TurnStartBlock,
)

# ContextVar for tool log capture (activated per-tool-call during agent loop)
_tool_log_buffer: ContextVar[list[str] | None] = ContextVar(
    "_tool_log_buffer", default=None
)


class ToolLogCaptureHandler(logging.Handler):
    """Logging handler that captures records into a ContextVar buffer.

    When ``_tool_log_buffer`` is set to a list (during tool execution),
    formatted log messages are appended to it.  When ``None`` (default),
    this handler is a no-op.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)

    def emit(self, record: logging.LogRecord) -> None:
        buf = _tool_log_buffer.get()
        if buf is not None:
            buf.append(self.format(record))

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 25


def _gen_id() -> str:
    """生成短 ID。"""
    return uuid4().hex[:12]


def _get_tool_calls(msg: Message) -> list[ToolUseBlock]:
    """从 Message.blocks 中提取 ToolUseBlock 列表。"""
    return [b for b in msg.blocks if isinstance(b, ToolUseBlock)]


def _get_tool_capture_enabled(agent: Agent) -> bool:
    """Check if tool log capture is enabled via any registered tool source's LogStore."""
    entries = getattr(agent.tools, '_entries', None)
    if not entries:
        return False
    for entry in entries.values():
        log_store = getattr(entry.source, "log_store", None)
        if log_store is not None:
            return log_store.tool_capture_enabled
    return False


def _event_listeners(agent: Agent) -> list[Callable[[StreamEvent], Any]]:
    listeners = getattr(agent, "_event_listeners", None)
    if listeners is None:
        listeners = []
        object.__setattr__(agent, "_event_listeners", listeners)
    return listeners


async def _emit_event(agent: Agent, event: StreamEvent) -> None:
    for callback in list(_event_listeners(agent)):
        result = callback(event)
        if inspect.isawaitable(result):
            await result


async def agent_run(
    self: Agent,
    input_stream: AsyncIterator[Message],
    stream: bool = True,
    check_pending: Callable[[], bool] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Run the agent conversation loop, consuming input messages and yielding output events."""
    _partial_text: list[str] = []
    try:
        async for msg in input_stream:
            self.context.messages.append(msg)

            # 提取 TurnStartBlock — 有则触发处理，无则只存储
            turn_start = None
            for b in msg.blocks:
                if isinstance(b, TurnStartBlock):
                    turn_start = b
                    break
            if turn_start is None:
                continue

            turn_id = turn_start.turn_id or _gen_id()

            # 计算用户消息文本长度用于日志
            text_len = sum(len(b.text) for b in msg.blocks if isinstance(b, TextBlock))
            logger.info("User message received (%d chars)", text_len)

            tool_round = 0
            _partial_text.clear()

            while True:
                # --- response_start ---
                msg_id = _gen_id()
                response_start_ts = time.time()
                model = self.model

                yield StreamEvent(
                    type="response_start",
                    response=Response(
                        message=Message(
                            role="assistant",
                            id=msg_id,
                            model=model,
                            timestamp=response_start_ts,
                        ),
                    ),
                )

                # --- LLM step ---
                response = None
                got_error = False
                async for event in agent_step(self, stream=stream):
                    if event.type == "text_delta" and event.text:
                        _partial_text.append(event.text)
                    yield event
                    if event.type == "response_done":
                        response = event.response
                    elif event.type == "error":
                        got_error = True
                        break

                if got_error:
                    _partial_text.clear()
                    break

                if response is None:
                    _partial_text.clear()
                    yield StreamEvent(
                        type="error",
                        error="No response_done event received from LLM",
                    )
                    break

                # --- 设置 assistant Message 元数据 ---
                response.message.id = msg_id
                response.message.timestamp = response_start_ts
                response.message.model = model
                response.message.duration = time.time() - response_start_ts
                response.message.input_tokens = response.usage.get("input_tokens", 0)
                response.message.output_tokens = response.usage.get("output_tokens", 0)

                # Update token usage
                self.context.update_usage(response.usage)

                # Add assistant message to history
                self.context.messages.append(response.message)
                _partial_text.clear()  # 已提交到 message，清空

                tool_calls = _get_tool_calls(response.message)
                logger.info("LLM stop_reason=%s, tool_calls=%d",
                            response.stop_reason, len(tool_calls))

                if tool_calls:
                    if tool_round >= MAX_TOOL_ROUNDS:
                        logger.warning(
                            "Tool call limit reached (%d rounds). "
                            "Injecting summary request.", MAX_TOOL_ROUNDS,
                        )
                        self.context.messages.append(Message(
                            role="user",
                            blocks=[TextBlock(
                                text="[System] Tool call limit reached. "
                                     "Summarize your progress and what remains to be done.",
                            )],
                        ))
                        async for event in agent_step(self, stream=stream):
                            yield event
                            if event.type == "response_done" and event.response:
                                self.context.update_usage(event.response.usage)
                                self.context.messages.append(event.response.message)
                        break

                    tool_round += 1

                    if response.stop_reason != "tool_use":
                        logger.warning(
                            "stop_reason=%s but %d tool_calls found in response, "
                            "executing tools anyway",
                            response.stop_reason, len(tool_calls),
                        )

                    # Execute tool calls
                    capture = _get_tool_capture_enabled(self)
                    tool_results: list[ToolResultBlock] = []
                    for block in tool_calls:
                        logger.info("Executing tool: %s", block.name)
                        args_str = str(block.input)
                        if len(args_str) > 200:
                            args_str = args_str[:200] + f"...({len(args_str)} chars total)"
                        logger.debug("Tool args: %s", args_str)

                        yield StreamEvent(type="tool_exec_start", tool_call=block, timestamp=time.time())
                        if capture:
                            buf: list[str] = []
                            token = _tool_log_buffer.set(buf)
                            try:
                                result_block = await self.tools.dispatch(block)
                            finally:
                                _tool_log_buffer.reset(token)
                            if buf:
                                suffix = "\n\n[Tool Logs]\n" + "\n".join(buf)
                                result_block.content += suffix
                        else:
                            result_block = await self.tools.dispatch(block)

                        logger.info("Tool %s result: %s (%d chars)",
                                    block.name,
                                    "error" if result_block.is_error else "ok",
                                    len(result_block.content))
                        logger.debug("Tool result content: %.200s", result_block.content)
                        tool_results.append(result_block)
                        yield StreamEvent(
                            type="tool_exec_end",
                            tool_call=result_block,
                            timestamp=time.time(),
                        )

                    if tool_results:
                        self.context.messages.append(
                            Message(role="user", blocks=list(tool_results))
                        )

                    # Natural checkpoint: check for pending user input
                    if check_pending and check_pending():
                        logger.info("Pending input detected at tool round checkpoint, ending turn early")
                        break
                else:
                    break

            # --- Turn 结束 ---
            yield StreamEvent(type="turn_done", turn_id=turn_id)

    finally:
        # --- 中断清理 ---
        # 提交部分文本（正常退出时 _partial_text 为空，no-op）
        if _partial_text:
            self.context.messages.append(Message(
                role="assistant",
                blocks=[TextBlock(text="".join(_partial_text) + "\n\n[interrupted]")],
            ))
async def agent_step(
    self: Agent, stream: bool = True
) -> AsyncIterator[StreamEvent]:
    """Execute a single LLM call, yielding streaming events."""
    tools = self.tools.get_tools()
    prompts = self.context.prepare_prompts()
    messages = self.context.prepare_messages()
    async for event in self.llm.send(
        messages, tools, prompts=prompts, stream=stream,
    ):
        yield event


@mutobj.impl(Agent.handle_tool_calls)
async def agent_handle_tool_calls(
    self: Agent, tool_calls: list[ToolUseBlock]
) -> list[ToolResultBlock]:
    """Dispatch tool calls through the tool set and return result blocks."""
    results: list[ToolResultBlock] = []
    for block in tool_calls:
        results.append(await self.tools.dispatch(block))
    return results


@mutobj.impl(Agent.subscribe)
def agent_subscribe(
    self: Agent, callback: Callable[[StreamEvent], Any]
) -> CancelFn:
    listeners = _event_listeners(self)
    listeners.append(callback)

    def _dispose() -> None:
        if callback in listeners:
            listeners.remove(callback)

    return _dispose


@mutobj.impl(Agent.is_busy)
def agent_is_busy(self: Agent) -> bool:
    task = getattr(self, "_current_task", None)
    return bool(task is not None and not task.done())


@mutobj.impl(Agent.submit)
async def agent_submit(self: Agent, text: str) -> None:
    if agent_is_busy(self):
        raise RuntimeError("Agent is busy")

    turn_id = _gen_id()
    logger.info("Scheduling submit turn %s (%d chars)", turn_id, len(text))

    async def _single_input() -> AsyncIterator[Message]:
        yield Message(
            role="user",
            blocks=[TurnStartBlock(turn_id=turn_id), TextBlock(text=text)],
        )

    async def _drive() -> None:
        saw_turn_done = False
        try:
            logger.info("Submit task started (turn=%s)", turn_id)
            async for event in agent_run(self, _single_input()):
                if event.type == "turn_done":
                    saw_turn_done = True
                await _emit_event(self, event)
            logger.info("Submit task finished (turn=%s)", turn_id)
        except asyncio.CancelledError:
            logger.info("Submit task cancelled (turn=%s)", turn_id)
            if not saw_turn_done:
                await _emit_event(
                    self,
                    StreamEvent(
                        type="turn_done",
                        turn_id=turn_id,
                        timestamp=time.time(),
                    ),
                )
            raise
        except Exception as exc:
            logger.exception("Agent submit task failed")
            await _emit_event(
                self,
                StreamEvent(
                    type="error",
                    error=str(exc),
                    turn_id=turn_id,
                    timestamp=time.time(),
                ),
            )
            if not saw_turn_done:
                await _emit_event(
                    self,
                    StreamEvent(
                        type="turn_done",
                        turn_id=turn_id,
                        timestamp=time.time(),
                    ),
                )

    task = asyncio.create_task(_drive())
    object.__setattr__(self, "_current_task", task)

    def _cleanup(done: asyncio.Task[None]) -> None:
        current = getattr(self, "_current_task", None)
        if current is done:
            object.__setattr__(self, "_current_task", None)
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Agent submit task crashed")

    task.add_done_callback(_cleanup)


@mutobj.impl(Agent.cancel)
def agent_cancel(self: Agent) -> bool:
    task = getattr(self, "_current_task", None)
    if task is None or task.done():
        logger.info("Cancel ignored: no running task")
        return False
    logger.info("Cancelling current submit task")
    task.cancel()
    return True
