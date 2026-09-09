"""API regression tests: bounded uploads and the existing shared chat queue."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiohttp import FormData, web
from aiohttp.test_utils import AioHTTPTestCase

from astrbot_plugin_arena_image import openai_proxy
from astrbot_plugin_arena_image.bridge_client import BridgeError
from .test_openai_proxy import FakePlugin, PNG


HEADERS = {"Authorization": "Bearer test-key"}
BOUNDARY = "fixture-boundary"


def multipart_body(fields):
    body = bytearray()
    for name, value in fields:
        body.extend(
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        body.extend(value)
        body.extend(b"\r\n")
    body.extend(f"--{BOUNDARY}--\r\n".encode())
    return bytes(body)


async def chunked(body):
    for offset in range(0, len(body), 257):
        yield body[offset : offset + 257]
        await asyncio.sleep(0)


class OpenAIProxyLimitsTest(AioHTTPTestCase):
    async def get_application(self):
        self.plugin = FakePlugin()
        self.plugin.config["max_queue_depth"] = 1
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        image = Path(directory.name) / "fixture.png"
        image.write_bytes(PNG)
        self.plugin._materialize_output = AsyncMock(return_value=image)
        self.proxy = openai_proxy.OpenAIProxyServer(self.plugin)
        self.proxy._web = web
        app = web.Application(client_max_size=4096)
        app.router.add_post("/v1/images/generations", self.proxy._handle_generations)
        app.router.add_post("/v1/images/edits", self.proxy._handle_edits)
        return app

    async def post_edit(self, fields, *, streaming=False):
        body = multipart_body(fields)
        return await self.client.post(
            "/v1/images/edits",
            headers={**HEADERS, "Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
            data=chunked(body) if streaming else body,
        )

    async def test_json_overflow_remains_413_instead_of_upstream_502(self):
        response = await self.client.post(
            "/v1/images/generations", headers=HEADERS, json={"prompt": "x" * 5000}
        )
        self.assertEqual(response.status, 413)
        self.assertEqual((await response.json())["error"]["code"], "payload_too_large")

    async def test_known_multipart_content_length_is_bounded(self):
        response = await self.post_edit([("prompt", b"x" * 5000), ("image", PNG)])
        self.assertEqual(response.status, 413)
        self.assertFalse(hasattr(self.plugin.client, "last"))

    async def test_chunked_multipart_cannot_skip_the_total_limit(self):
        response = await self.post_edit(
            [("ignored", b"a" * 2100), ("ignored-again", b"b" * 2100), ("image", PNG)],
            streaming=True,
        )
        self.assertEqual(response.status, 413)
        self.assertFalse(hasattr(self.plugin.client, "last"))

    async def test_chunked_valid_edit_and_skipped_fields_still_work(self):
        response = await self.post_edit(
            [("ignored", b"a" * 700), ("prompt", b"edit"), ("image", PNG)],
            streaming=True,
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(self.plugin.client.last[1], "edit")
        self.assertEqual(len(self.plugin.client.last[2]), 1)
        self.assertEqual(self.plugin._active_generations, 0)

    async def test_text_field_has_its_own_limit(self):
        with patch.object(openai_proxy, "MAX_FORM_TEXT_BYTES", 16):
            response = await self.post_edit([("prompt", b"x" * 17), ("image", PNG)])
        self.assertEqual(response.status, 413)

    async def test_each_uploaded_image_is_bounded(self):
        self.plugin._input_max_bytes = lambda: 8
        response = await self.post_edit([("prompt", b"edit"), ("image", PNG)])
        self.assertEqual(response.status, 413)

    async def test_image_count_is_bounded(self):
        self.plugin.config["max_input_images"] = 1
        response = await self.post_edit([("image", PNG), ("image[]", PNG)])
        self.assertEqual(response.status, 400)

    async def test_bad_json_image_shape_is_a_client_error(self):
        response = await self.client.post(
            "/v1/images/edits",
            headers=HEADERS,
            json={"prompt": "edit", "images": {"unexpected": "shape"}},
        )
        self.assertEqual(response.status, 400)

    async def test_standard_file_upload_keeps_image_data(self):
        form = FormData()
        form.add_field("prompt", "edit")
        form.add_field("image", PNG, filename="fixture.png", content_type="image/png")
        response = await self.client.post("/v1/images/edits", headers=HEADERS, data=form)
        self.assertEqual(response.status, 200)
        self.assertTrue(self.plugin.client.last[2][0].startswith("data:image/png;base64,"))

    async def test_full_chat_queue_rejects_api_without_calling_transport(self):
        self.plugin._active_generations = 1
        response = await self.client.post(
            "/v1/images/generations", headers=HEADERS, json={"prompt": "fixture"}
        )
        self.assertEqual(response.status, 429)
        self.assertEqual(self.plugin._active_generations, 1)
        self.assertFalse(hasattr(self.plugin.client, "last"))

    async def test_running_api_is_counted_and_rejects_another_request(self):
        started, finish = asyncio.Event(), asyncio.Event()
        complete = self.plugin.client.complete

        async def blocked(**kwargs):
            started.set()
            await finish.wait()
            return await complete(**kwargs)

        self.plugin.client.complete = blocked
        first = asyncio.create_task(self.proxy._generate_images("image-model", "first", [], 1))
        try:
            await asyncio.wait_for(started.wait(), 3)
            self.assertEqual(self.plugin._active_generations, 1)
            response = await self.client.post(
                "/v1/images/generations", headers=HEADERS, json={"prompt": "second"}
            )
            self.assertEqual(response.status, 429)
        finally:
            finish.set()
            await first
        self.assertEqual(self.plugin._active_generations, 0)
        self.assertGreaterEqual(self.plugin._last_generation_seconds, 0)

    async def test_cancelled_queue_waiter_releases_its_shared_slot(self):
        await self.plugin._generation_lock.acquire()
        task = asyncio.create_task(self.proxy._generate_images("image-model", "fixture", [], 1))
        try:
            await asyncio.sleep(0)
            self.assertEqual(self.plugin._active_generations, 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(self.plugin._active_generations, 0)
        finally:
            self.plugin._generation_lock.release()

    async def test_upstream_failure_releases_its_shared_slot(self):
        self.plugin.client.complete = AsyncMock(side_effect=BridgeError("fixture failure"))
        with self.assertRaises(BridgeError):
            await self.proxy._generate_images("image-model", "fixture", [], 1)
        self.assertEqual(self.plugin._active_generations, 0)

    async def test_live_old_model_is_not_overridden_by_a_missing_alias_target(self):
        self.plugin._fetch_models = AsyncMock(return_value=[{"id": "mona-lisa-alpha"}])
        self.assertEqual(await self.proxy._resolve_model("mona-lisa-alpha"), "mona-lisa-alpha")

    async def test_missing_alias_is_not_exposed_as_an_available_model(self):
        self.plugin._fetch_models = AsyncMock(return_value=[{"id": "image-model"}])
        with self.assertRaises(BridgeError):
            await self.proxy._resolve_model("mona-lisa-alpha")

    async def test_disabled_proxy_opens_no_listener(self):
        self.plugin.config["openai_proxy_enabled"] = False
        proxy = openai_proxy.OpenAIProxyServer(self.plugin)
        await proxy.start()
        self.assertIsNone(proxy._runner)
        await proxy.stop()

    async def test_lifecycle_can_start_stop_and_start_again(self):
        proxy = openai_proxy.OpenAIProxyServer(self.plugin)
        proxy._port_value = lambda: 0
        try:
            await proxy.start()
            self.assertIsNotNone(proxy._runner)
            await proxy.stop()
            self.assertIsNone(proxy._runner)
            await proxy.start()
            self.assertIsNotNone(proxy._runner)
        finally:
            await proxy.stop()
