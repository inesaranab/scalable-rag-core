"""The sandbox runner: executes code in a killable child, captures stdout."""

import httpx

from services.sandbox.runner import app, execute_in_child


async def _post(payload: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://sandbox.test"
    ) as client:
        return await client.post("/execute", json=payload)


async def test_prints_come_back_as_output():
    response = await _post({"code": "print(21 * 2)"})

    assert response.status_code == 200
    assert response.json() == {"status": "success", "output": "42\n"}


async def test_an_exception_comes_back_as_an_error_not_a_crash():
    response = await _post({"code": "1 / 0"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert "division" in body["output"]


async def test_an_infinite_loop_is_killed_at_the_timeout():
    response = await _post({"code": "while True: pass", "timeout": 1})

    assert response.status_code == 408
    assert "timed out" in response.json()["output"]


async def test_code_larger_than_the_cap_is_refused():
    response = await _post({"code": "x = 1\n" * 10_000})

    assert response.status_code == 413


def test_the_child_reports_through_the_queue():
    import multiprocessing

    queue = multiprocessing.Queue()
    execute_in_child("print('hola')", queue)

    assert queue.get() == {"status": "success", "output": "hola\n"}
