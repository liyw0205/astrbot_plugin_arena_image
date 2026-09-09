"""Optional OpenAI-compatible image API exposed by the AstrBot plugin.

The proxy deliberately stays small: model discovery and image generation are
delegated to the plugin's existing transport client, so bridge and direct-CDP
mode behave identically.  Images are returned as ``b64_json`` because signed
Arena URLs are short-lived and are usually not reachable by the API caller.
"""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import time
from typing import Any

from .bridge_client import (
    BridgeError,
    data_uri_from_base64,
    decode_image_value,
    guess_image_mime,
    image_urls,
    response_text,
)


class OpenAIProxyServer:
    """Lifecycle-managed aiohttp server for the three OpenAI image endpoints."""

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin
        self._runner: Any = None
        self._site: Any = None
        self._web: Any = None
        self._host = ""
        self._port = 0

    @property
    def address(self) -> str:
        if not self._host or not self._port:
            return ""
        return f"http://{self._host}:{self._port}/v1"

    async def start(self) -> None:
        if self._runner is not None or not self._enabled():
            return
        try:
            from aiohttp import web
        except ImportError as exc:  # pragma: no cover - dependency is installed in production
            raise RuntimeError("启用 OpenAI 中转需要安装 aiohttp") from exc

        self._web = web
        host = str(self.plugin.config.get("openai_proxy_host") or "127.0.0.1").strip()
        port = self._port_value()
        # JSON Data URIs expand a binary image by roughly 4/3; leave room for
        # several configured reference images plus multipart/JSON overhead.
        max_body = max(
            16 * (1 << 20),
            self.plugin._input_max_bytes() * self._max_input_images() * 2 + (1 << 20),
        )
        app = web.Application(client_max_size=max_body)
        for path in ("/v1/models", "/models"):
            app.router.add_get(path, self._handle_models)
        for path in ("/v1/images/generations", "/images/generations"):
            app.router.add_post(path, self._handle_generations)
        for path in ("/v1/images/edits", "/images/edits"):
            app.router.add_post(path, self._handle_edits)
        app.router.add_options("/{tail:.*}", self._handle_options)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        try:
            site = web.TCPSite(runner, host, port)
            await site.start()
        except Exception:
            await runner.cleanup()
            raise
        self._runner = runner
        self._site = site
        self._host = host
        self._port = port

    async def stop(self) -> None:
        runner, self._runner = self._runner, None
        self._site = None
        self._host = ""
        self._port = 0
        if runner is not None:
            await runner.cleanup()

    def _enabled(self) -> bool:
        value = self.plugin.config.get("openai_proxy_enabled", False)
        return value is True or str(value).strip().casefold() in {"1", "true", "yes", "on"}

    def _port_value(self) -> int:
        try:
            port = int(self.plugin.config.get("openai_proxy_port", 18081))
        except (TypeError, ValueError):
            port = 18081
        return max(1, min(65535, port))

    def _api_key(self) -> str:
        return str(self.plugin.config.get("openai_proxy_api_key") or "").strip()

    def _is_loopback_host(self) -> bool:
        return str(self.plugin.config.get("openai_proxy_host") or "127.0.0.1").strip().casefold() in {
            "127.0.0.1",
            "localhost",
            "::1",
        }

    def _auth_error(self):
        return self._web.json_response(
            {
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
            status=401,
            headers={**self._cors_headers(), "WWW-Authenticate": "Bearer"},
        )

    def _authorize(self, request: Any):
        configured = self._api_key()
        # A blank key is useful for a loopback-only local client, but never
        # silently exposes an unauthenticated listener on a public interface.
        if not configured:
            if self._is_loopback_host():
                return None
            return self._web.json_response(
                {
                    "error": {
                        "message": "openai_proxy_api_key is required when the proxy is not loopback-only",
                        "type": "configuration_error",
                        "code": "missing_api_key",
                    }
                },
                status=503,
                headers=self._cors_headers(),
            )
        supplied = request.headers.get("Authorization", "")
        if supplied.casefold().startswith("bearer "):
            supplied = supplied[7:].strip()
        if not supplied:
            supplied = request.headers.get("X-API-Key", "").strip()
        if not secrets.compare_digest(supplied, configured):
            return self._auth_error()
        return None

    async def _handle_options(self, request: Any):
        return self._web.Response(status=204, headers=self._cors_headers())

    async def _handle_models(self, request: Any):
        if (error := self._authorize(request)) is not None:
            return error
        try:
            models = await self.plugin._fetch_models()
            data = []
            for model in models:
                model_id = self.plugin._model_id(model)
                if not model_id:
                    continue
                created = model.get("created")
                try:
                    created = int(created or 0)
                except (TypeError, ValueError):
                    created = 0
                data.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": created or int(time.time()),
                        "owned_by": str(
                            model.get("owned_by") or model.get("organization") or "arena"
                        ),
                        "input_image": bool(model.get("input_image")),
                        "output_image": bool(model.get("output_image", True)),
                    }
                )
            return self._json({"object": "list", "data": data})
        except Exception as exc:
            return self._error_response(exc)

    async def _handle_generations(self, request: Any):
        if (error := self._authorize(request)) is not None:
            return error
        try:
            body = await self._json_body(request)
            prompt = str(body.get("prompt") or "").strip()
            if not prompt:
                return self._bad_request("prompt is required")
            model = await self._resolve_model(str(body.get("model") or ""))
            count = self._requested_count(body.get("n"))
            data = await self._generate_images(model, prompt, [], count)
            return self._json({"created": int(time.time()), "data": data})
        except Exception as exc:
            return self._error_response(exc)

    async def _handle_edits(self, request: Any):
        if (error := self._authorize(request)) is not None:
            return error
        try:
            prompt, model_name, count, images = await self._parse_edit_request(request)
            if not images:
                return self._bad_request("image is required")
            if not prompt:
                prompt = "根据参考图生成一张更精致的图片"
            model = await self._resolve_model(model_name)
            data = await self._generate_images(model, prompt, images, count)
            return self._json({"created": int(time.time()), "data": data})
        except Exception as exc:
            return self._error_response(exc)

    async def _json_body(self, request: Any) -> dict[str, Any]:
        try:
            payload = await request.json()
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("请求体必须是 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    async def _parse_edit_request(self, request: Any) -> tuple[str, str, int, list[str]]:
        if str(request.content_type or "").casefold() == "application/json":
            body = await self._json_body(request)
            prompt = str(body.get("prompt") or "").strip()
            model = str(body.get("model") or "").strip()
            count = self._requested_count(body.get("n"))
            values = body.get("images", body.get("image", []))
            if isinstance(values, str):
                values = [values]
            images = self._normalize_images(values or [])
            return prompt, model, count, images

        if not str(request.content_type or "").startswith("multipart/"):
            raise ValueError("图生图接口需要 multipart/form-data 或 JSON")
        reader = await request.multipart()
        prompt = ""
        model = ""
        count_value: Any = 1
        images: list[str] = []
        max_bytes = self.plugin._input_max_bytes()
        while True:
            part = await reader.next()
            if part is None:
                break
            name = str(part.name or "").casefold()
            if name in {"prompt", "model", "n"}:
                value = (await part.text()).strip()
                if name == "prompt":
                    prompt = value
                elif name == "model":
                    model = value
                else:
                    count_value = value
                continue
            if name not in {"image", "images", "image[]"}:
                continue
            raw = await self._read_part(part, max_bytes)
            if not raw:
                continue
            try:
                checked_raw, mime = decode_image_value(
                    raw,
                    source=part.filename or "upload.png",
                    mime_type=part.headers.get("Content-Type"),
                    max_bytes=max_bytes,
                )
            except BridgeError as exc:
                raise ValueError(str(exc)) from exc
            data_uri = "data:{mime};base64,{encoded}".format(
                mime=mime,
                encoded=base64.b64encode(checked_raw).decode("ascii"),
            )
            images.append(data_uri)
            if len(images) > self._max_input_images():
                raise ValueError(f"最多只能上传 {self._max_input_images()} 张图片")
        return prompt, model, self._requested_count(count_value), images

    def _max_input_images(self) -> int:
        try:
            value = int(self.plugin.config.get("max_input_images", 4) or 4)
        except (TypeError, ValueError):
            value = 4
        return max(1, min(8, value))

    def _normalize_images(self, values: list[Any]) -> list[str]:
        if len(values) > self._max_input_images():
            raise ValueError(f"最多只能上传 {self._max_input_images()} 张图片")
        normalized: list[str] = []
        max_bytes = self.plugin._input_max_bytes()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            if text.casefold().startswith(("http://", "https://")):
                normalized.append(text)
                continue
            try:
                raw, mime = decode_image_value(text, max_bytes=max_bytes)
            except BridgeError as exc:
                raise ValueError(str(exc)) from exc
            normalized.append(
                data_uri_from_base64(raw, mime_type=mime, max_bytes=max_bytes)
            )
        return normalized

    @staticmethod
    async def _read_part(part: Any, max_bytes: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await part.read_chunk(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"图片超过大小限制（{max_bytes // 1024 // 1024} MB）")
            chunks.append(chunk)
        return b"".join(chunks)

    def _requested_count(self, value: Any) -> int:
        try:
            requested = int(value or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError("n must be an integer") from exc
        maximum = max(1, min(4, int(self.plugin.config.get("max_output_images", 1) or 1)))
        if requested < 1 or requested > maximum:
            raise ValueError(f"n must be between 1 and {maximum}")
        return requested

    async def _resolve_model(self, requested: str) -> str:
        models = await self.plugin._fetch_models()
        if not models:
            raise BridgeError("没有可用的画图模型")
        wanted = str(requested or "").strip().casefold()
        by_id = {self.plugin._model_id(model).casefold(): self.plugin._model_id(model) for model in models}
        if wanted:
            if wanted not in by_id:
                raise BridgeError(f"模型不存在：{requested}")
            return by_id[wanted]
        selected = self.plugin._selected_models.get("__global__", "")
        default = str(self.plugin.config.get("default_model") or "").strip()
        for candidate in (selected, default):
            if candidate.casefold() in by_id:
                return by_id[candidate.casefold()]
        return self.plugin._model_id(models[0])

    async def _generate_images(
        self,
        model: str,
        prompt: str,
        images: list[str],
        count: int,
    ) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        async with self.plugin._generation_lock:
            while len(result) < count:
                response = await self.plugin._client().complete(
                    model=model,
                    prompt=prompt,
                    images=images or None,
                )
                candidates = image_urls(response)
                if not candidates:
                    text = response_text(response).strip()
                    raise BridgeError(text or "模型没有返回图片")
                for candidate in candidates:
                    path = await self.plugin._materialize_output(candidate)
                    raw = path.read_bytes()
                    result.append(
                        {
                            "b64_json": base64.b64encode(raw).decode("ascii"),
                            "mime_type": guess_image_mime(path.name, raw),
                        }
                    )
                    if len(result) >= count:
                        break
        self.plugin._prune_outputs()
        return result

    def _json(self, payload: Any):
        return self._web.json_response(payload, headers=self._cors_headers())

    def _bad_request(self, message: str):
        return self._web.json_response(
            {"error": {"message": message, "type": "invalid_request_error", "code": "bad_request"}},
            status=400,
            headers=self._cors_headers(),
        )

    def _error_response(self, exc: Exception):
        if isinstance(exc, ValueError):
            status = 400
            code = "bad_request"
        elif isinstance(exc, BridgeError) and exc.is_rate_limited:
            status = 429
            code = "rate_limit_exceeded"
        elif isinstance(exc, BridgeError) and exc.status_code in {401, 403}:
            status = 502
            code = "upstream_auth_error"
        else:
            status = 502
            code = "upstream_error"
        message = str(exc).strip() or type(exc).__name__
        return self._web.json_response(
            {"error": {"message": message[:1000], "type": "api_error", "code": code}},
            status=status,
            headers=self._cors_headers(),
        )

    @staticmethod
    def _cors_headers() -> dict[str, str]:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, X-API-Key",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        }
