"""The name catalog is discoverable, read-only, and not a model picker."""

from __future__ import annotations

import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from astrbot_plugin_arena_image import arena_direct, bridge_client
from tests.test_arena_direct_transport import (
    DirectTransportCase,
    FakePage,
    _filler_models,
    _image_model,
)
from tests.test_arena_image_hardening import FakeEvent, _collect, _make_plugin


def entry(name, *, display=None, selectable=True, upstream=True):
    return {
        "id": name,
        "display_name": display or name,
        "user_selectable": upstream,
        "selectable": selectable,
    }


def gpt_catalog():
    return {
        "complete": True,
        "models": [
            entry("gpt-image-1"),
            entry("gpt-image-1-high-fidelity"),
            *[entry("gpt-image-2 (medium)") for _ in range(4)],
            entry("gpt-image-2 (medium)", selectable=False, upstream=False),
            entry("gpt-image-1.5-high-fidelity", selectable=False, upstream=False),
            entry(
                "chatgpt-image-latest-high-fidelity (20251216)", selectable=False, upstream=False
            ),
            entry("gpt-image-2.5-flare", selectable=False, upstream=False),
            entry("lina-alpha", display="gpt-image-2.5-flare", selectable=False, upstream=False),
            entry("gpt-image-2.5-sunburst", selectable=False, upstream=False),
            entry(
                "lina-f-alpha", display="gpt-image-2.5-sunburst", selectable=False, upstream=False
            ),
        ],
    }


class DirectNameCatalogTests(DirectTransportCase):
    def test_hidden_names_are_visible_only_to_the_read_only_catalog(self):
        hidden = _image_model(
            "gpt-hidden", "hidden-id", userSelectable=False, displayName="gpt-image-2.5-flare"
        )
        visible = _image_model("gpt-image-1", "visible-id", userSelectable=True)
        text_only = {
            "id": "chat-id",
            "publicName": "gpt-chat",
            "capabilities": {"outputCapabilities": {"text": True}},
        }
        page = FakePage(models=_filler_models() + [hidden, visible, text_only])
        client = self.client(page)
        catalog = self.run_async(client.model_name_catalog())
        by_id = {row["id"]: row for row in catalog["models"]}
        self.assertTrue(catalog["complete"])
        self.assertEqual(by_id["gpt-hidden"]["display_name"], "gpt-image-2.5-flare")
        self.assertFalse(by_id["gpt-hidden"]["user_selectable"])
        self.assertFalse(by_id["gpt-hidden"]["selectable"])
        self.assertTrue(by_id["gpt-image-1"]["selectable"])
        self.assertNotIn("gpt-chat", by_id)
        selectable = self.run_async(client.list_models())
        self.assertNotIn("gpt-hidden", [row["id"] for row in selectable])
        self.assertEqual(client._variants(arena_direct._MODELS, "gpt-hidden"), [])
        self.assertEqual(page.requests, [])
        self.assertEqual(page.actions, [])
        self.assertFalse((self.data_dir / "arena_model_health.json").exists())

    def test_local_stealth_setting_does_not_masquerade_as_upstream_restriction(self):
        row = _image_model("unnamed-fixture", "fixture-id", organization=None, userSelectable=True)
        page = FakePage(models=_filler_models() + [row])
        client = self.client(page, allow_stealth_models=False)
        catalog = self.run_async(client.model_name_catalog())
        found = next(item for item in catalog["models"] if item["id"] == "unnamed-fixture")
        self.assertTrue(found["user_selectable"])
        self.assertFalse(found["selectable"])
        self.assertFalse(client._keep_model(row))

    def test_explicit_catalog_request_refreshes_the_model_table(self):
        page = FakePage(models=_filler_models())
        client = self.client(page)
        self.run_async(client.model_name_catalog())
        page._models.append(_image_model("gpt-new", "new-id", userSelectable=False))
        catalog = self.run_async(client.model_name_catalog())
        self.assertIn("gpt-new", [row["id"] for row in catalog["models"]])
        self.assertEqual(page.table_reads, 2)
        self.assertEqual(page.requests, [])


class NameCatalogCommandTests(unittest.TestCase):
    def plugin(self, payload=None, config=None):
        main, plugin = _make_plugin(self, config)
        client = SimpleNamespace(
            model_name_catalog=AsyncMock(return_value=payload or gpt_catalog()),
            complete=AsyncMock(),
            model_health=AsyncMock(),
        )
        plugin._client = lambda: client
        return main, plugin, client

    def text(self, plugin, keyword="GPT"):
        return asyncio.run(plugin._model_name_list_text(keyword))

    def test_all_seven_gpt_names_are_together_with_explicit_statuses(self):
        _, plugin, client = self.plugin()
        text = self.text(plugin)
        self.assertIn("7 个名称", text)
        self.assertIn("• gpt-image-1 — 可选", text)
        self.assertIn("• gpt-image-1-high-fidelity — 可选", text)
        self.assertIn("• gpt-image-2 (medium) — 可选（4 个变体）", text)
        self.assertIn("• gpt-image-1.5-high-fidelity — 上游未开放", text)
        self.assertIn("• gpt-image-2.5-flare — 上游未开放", text)
        self.assertIn("• gpt-image-2.5-sunburst — 上游未开放", text)
        self.assertIn("chatgpt-image-latest-high-fidelity (20251216) — 上游未开放", text)
        self.assertEqual(text.count("• gpt-image-2.5-flare"), 1)
        self.assertIn("上游名称对照：lina-alpha", text)
        self.assertIn("上游名称对照：lina-f-alpha", text)
        self.assertIn("不代表已验证出图成功", text)
        self.assertIn("本表没有切换编号", text)
        self.assertNotIn("✅", text)
        client.complete.assert_not_awaited()
        client.model_health.assert_not_awaited()

    def test_name_lookup_keeps_selection_presets_and_data_files_unchanged(self):
        _, plugin, _ = self.plugin()
        plugin._selected_models = {"__global__": "original-model"}
        plugin._presets = {"saved": {"prompt": "original prompt", "model": "original-model"}}
        before_selection = copy.deepcopy(plugin._selected_models)
        before_presets = copy.deepcopy(plugin._presets)
        before_files = {
            str(path): path.read_bytes() for path in plugin.data_dir.rglob("*") if path.is_file()
        }
        result = _collect(plugin.list_model_names(FakeEvent(), "GPT"))
        self.assertEqual(len(result), 1)
        self.assertIn("gpt-image-2.5-flare", result[0].text)
        self.assertEqual(plugin._selected_models, before_selection)
        self.assertEqual(plugin._presets, before_presets)
        after_files = {
            str(path): path.read_bytes() for path in plugin.data_dir.rglob("*") if path.is_file()
        }
        self.assertEqual(after_files, before_files)

    def test_query_alias_matches_the_whole_group_not_just_the_hidden_alias_row(self):
        payload = {
            "complete": True,
            "models": [
                entry("gpt-image-2.5-flare"),
                entry(
                    "lina-alpha", display="gpt-image-2.5-flare", selectable=False, upstream=False
                ),
            ],
        }
        _, plugin, _ = self.plugin(payload)
        text = self.text(plugin, "LINA")
        self.assertIn("1 个名称", text)
        self.assertIn("gpt-image-2.5-flare — 可选", text)
        self.assertIn("上游名称对照：lina-alpha", text)

    def test_display_name_does_not_replace_the_actual_selectable_request_name(self):
        payload = {
            "complete": True,
            "models": [entry("fixture-public-name", display="gpt-image-fixture")],
        }
        _, plugin, _ = self.plugin(payload)
        text = self.text(plugin)
        self.assertIn("可切换请求名：fixture-public-name", text)
        self.assertIn("上游名称对照：fixture-public-name", text)

    def test_local_restriction_and_upstream_restriction_have_different_labels(self):
        _, plugin, _ = self.plugin(
            {
                "complete": True,
                "models": [
                    entry("gpt-local", selectable=False),
                    entry("gpt-upstream", selectable=False, upstream=False),
                ],
            }
        )
        text = self.text(plugin)
        self.assertIn("gpt-local — 当前配置未放行", text)
        self.assertIn("gpt-upstream — 上游未开放", text)

    def test_inconsistent_flags_never_claim_a_hidden_model_is_selectable(self):
        _, plugin, _ = self.plugin(
            {
                "complete": True,
                "models": [entry("gpt-hidden", selectable=True, upstream=False)],
            }
        )
        self.assertIn("gpt-hidden — 上游未开放", self.text(plugin))

    def test_bridge_fallback_clearly_labels_its_filtered_scope(self):
        _, plugin = _make_plugin(self)
        client = SimpleNamespace(
            list_models=AsyncMock(
                return_value=[
                    {"id": "gpt-image-1", "output_image": True, "owned_by": "openai"},
                    {"id": "gpt-text", "output_image": False, "owned_by": "openai"},
                ]
            )
        )
        plugin._client = lambda: client
        text = self.text(plugin)
        self.assertIn("gpt-image-1 — 可选", text)
        self.assertNotIn("gpt-text", text)
        self.assertIn("当前 Bridge 仅返回可选模型", text)
        self.assertNotIn("gpt-image-2.5", text)

    def test_malformed_rows_are_skipped_and_empty_queries_default_to_gpt(self):
        _, plugin, _ = self.plugin(
            {
                "complete": True,
                "models": [None, {}, entry("gpt-image-1")],
            }
        )
        self.assertIn("1 个名称", self.text(plugin, " "))
        self.assertIn("0 个名称", self.text(plugin, "missing"))
        self.assertIn("未找到匹配名称", self.text(plugin, "missing"))

    def test_bad_catalog_shape_is_an_error_instead_of_a_false_empty_catalog(self):
        _, plugin, client = self.plugin()
        client.model_name_catalog.return_value = {"models": "invalid"}
        with self.assertRaises(bridge_client.BridgeError):
            self.text(plugin)
        result = _collect(plugin.list_model_names(FakeEvent(), "GPT"))
        self.assertIn("读取模型名称失败", result[0].text)

    def test_default_command_and_aliases_have_an_actual_registered_handler(self):
        main, plugin, _ = self.plugin()
        handler = main.ArenaImagePlugin.list_model_names
        self.assertEqual(handler.arena_command, "竞技场模型名称")
        self.assertIn("竞技场GPT模型", handler.arena_aliases)
        self.assertIn("竞技场gpt模型", handler.arena_aliases)
        self.assertNotEqual(getattr(handler, "arena_permission", None), "admin")
        result = _collect(plugin.list_model_names(FakeEvent()))
        self.assertIn("7 个名称", result[0].text)

    def test_configured_output_limit_is_respected(self):
        _, plugin, _ = self.plugin(config={"model_list_limit": 2})
        text = self.text(plugin)
        self.assertEqual(sum(line.startswith("• ") for line in text.splitlines()), 2)
        self.assertIn("其余 5 个已省略", text)

    def test_both_existing_lists_point_to_the_new_name_command(self):
        _, plugin = _make_plugin(self)
        plugin._fetch_models = AsyncMock(
            return_value=[
                {"id": "gpt-image-1", "owned_by": "openai", "output_image": True},
            ]
        )
        plugin._fetch_model_health = AsyncMock(return_value={})
        for stealth in (False, True):
            with self.subTest(stealth=stealth):
                text = asyncio.run(plugin._model_list_text(FakeEvent(), stealth=stealth))
                self.assertIn("/竞技场模型名称 GPT", text)
