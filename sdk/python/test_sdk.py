import asyncio
import os
import tempfile
import unittest

from carnot_sdk import CarnotClient, CarnotExtension


class DemoExt(CarnotExtension):
    timeout_ms = 5000

    def before_tool(self, tool, inp):
        return {"decision": "allow"}


class TestCarnotSdk(unittest.TestCase):
    def test_extension_init_uses_protocol_version_and_timeout(self):
        ext = DemoExt()
        resp = ext.handle_request({"jsonrpc": "2.0", "id": 1, "method": "ext/init", "params": {}})
        self.assertIsNotNone(resp)
        res = resp["result"]
        self.assertEqual(res["protocolVersion"], "carnot-ext/1")
        self.assertEqual(res["timeout_ms"], 5000)
        self.assertEqual(res["hooks"], ["before_tool"])

    def test_live_rpc_roundtrip_with_faux_provider(self):
        carnot_bin = os.environ.get("CARNOT_BIN")
        if not carnot_bin or not os.path.isfile(carnot_bin):
            self.skipTest("CARNOT_BIN not set")

        async def run():
            with tempfile.TemporaryDirectory() as td:
                async with CarnotClient(
                    binary=carnot_bin,
                    cwd=td,
                    provider="faux",
                    model="faux-1",
                    profile="minimal",
                    no_session=True,
                ) as client:
                    pong = await client.ping()
                    self.assertTrue(pong.get("pong"))
                    state = await client.get_state()
                    self.assertEqual(state["model"], "faux-1")
                    ttl_res = await client.set_cache_ttl("long")
                    self.assertEqual(ttl_res["cache_ttl"], "long")
                    state2 = await client.get_state()
                    self.assertEqual(state2["cache_ttl"], "long")
                    self.assertIn("prefix_retention_rate", state2)
                    metrics = await client.get_metrics()
                    self.assertIn("compacted_tokens_saved", metrics)
                    stats = await client.get_session_stats()
                    self.assertEqual(stats["cache_ttl"], "long")
                    self.assertIn("prefix_retention_rate", stats)
                    cmds = await client.get_commands()
                    self.assertIsInstance(cmds, dict)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
