"""测试 Server.views 路由过滤功能。

当多个 Server 实例监听不同端口但 View 使用相同路径时，
每个 Server 应该只路由到自己指定的 View。
"""

import pytest
import asyncio

import mutobj
from mutagent.net.server import Server, View, Request, Response


# ---------------------------------------------------------------------------
# 定义两个使用相同路径的 View
# ---------------------------------------------------------------------------


class ViewA(View):
    """View A - 路径 /api"""
    path = "/api"

    async def get(self, request: Request) -> Response:
        return Response(status_code=200, body=b"ViewA", headers={})


class ViewB(View):
    """View B - 同样路径 /api"""
    path = "/api"

    async def get(self, request: Request) -> Response:
        return Response(status_code=200, body=b"ViewB", headers={})


class ViewC(View):
    """View C - 不同路径 /other"""
    path = "/other"

    async def get(self, request: Request) -> Response:
        return Response(status_code=200, body=b"ViewC", headers={})


# ---------------------------------------------------------------------------
# 定义两个 Server，各自限制路由范围
# ---------------------------------------------------------------------------


class ServerA(Server):
    """Server A - 只路由到 ViewA"""
    views = (ViewA,)


class ServerB(Server):
    """Server B - 只路由到 ViewB 和 ViewC"""
    views = (ViewB, ViewC)


class ServerAll(Server):
    """Server All - 不限制，路由到所有 View（默认行为）"""
    pass


# ---------------------------------------------------------------------------
# 测试辅助函数
# ---------------------------------------------------------------------------


async def make_request(server: Server, path: str, method: str = "GET") -> Response | None:
    """模拟向 Server 发送请求并获取响应。"""
    responses = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            responses.append({"status": message["status"], "headers": message.get("headers", [])})
        elif message["type"] == "http.response.body":
            if responses:
                responses[-1]["body"] = message.get("body", b"")

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
        "query_string": b"",
    }

    await server.route(scope, receive, send)

    if responses:
        r = responses[0]
        return Response(status_code=r["status"], body=r.get("body", b""), headers={})
    return None


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_views_isolation():
    """测试 Server.views 限制路由范围。

    ServerA 设置 views=(ViewA,)，应该只能访问 ViewA。
    ServerB 设置 views=(ViewB, ViewC)，应该只能访问 ViewB 和 ViewC。
    """
    server_a = ServerA()
    server_b = ServerB()

    # ServerA 访问 /api 应该得到 ViewA 的响应
    resp_a = await make_request(server_a, "/api")
    assert resp_a is not None
    assert resp_a.status_code == 200
    assert resp_a.body == b"ViewA", f"ServerA /api should return ViewA, got {resp_a.body}"

    # ServerA 访问 /other 应该 404（ViewC 不在其 views 中）
    resp_a_other = await make_request(server_a, "/other")
    assert resp_a_other is not None
    assert resp_a_other.status_code == 404, f"ServerA /other should be 404, got {resp_a_other.status_code}"

    # ServerB 访问 /api 应该得到 ViewB 的响应（不是 ViewA）
    resp_b = await make_request(server_b, "/api")
    assert resp_b is not None
    assert resp_b.status_code == 200
    assert resp_b.body == b"ViewB", f"ServerB /api should return ViewB, got {resp_b.body}"

    # ServerB 访问 /other 应该得到 ViewC 的响应
    resp_b_other = await make_request(server_b, "/other")
    assert resp_b_other is not None
    assert resp_b_other.status_code == 200
    assert resp_b_other.body == b"ViewC", f"ServerB /other should return ViewC, got {resp_b_other.body}"


@pytest.mark.asyncio
async def test_server_without_views_routes_all():
    """测试没有设置 views 的 Server 路由到所有 View。"""
    server_all = ServerAll()

    # 访问 /api - 由于 ViewA 和 ViewB 都匹配，应该返回其中一个
    resp = await make_request(server_all, "/api")
    assert resp is not None
    assert resp.status_code == 200
    assert resp.body in (b"ViewA", b"ViewB"), f"ServerAll /api should return ViewA or ViewB, got {resp.body}"

    # 访问 /other
    resp_other = await make_request(server_all, "/other")
    assert resp_other is not None
    assert resp_other.status_code == 200
    assert resp_other.body == b"ViewC"


@pytest.mark.asyncio
async def test_same_path_different_servers():
    """测试相同路径在不同 Server 返回不同 View 的响应。

    这是核心隔离测试：
    - ServerA 的 /api 返回 "ViewA"
    - ServerB 的 /api 返回 "ViewB"
    """
    server_a = ServerA()
    server_b = ServerB()

    resp_a = await make_request(server_a, "/api")
    resp_b = await make_request(server_b, "/api")

    assert resp_a is not None and resp_b is not None
    assert resp_a.body == b"ViewA"
    assert resp_b.body == b"ViewB"
    assert resp_a.body != resp_b.body, "Same path on different servers should return different responses"
