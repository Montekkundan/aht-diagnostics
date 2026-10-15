import unittest
from pathlib import Path

from aht_diagnostics.jaxaht_bridge import resolve_heldout_config, select_heldout_member


class ResolveHeldoutConfigTests(unittest.TestCase):
    def test_relative_path_is_joined_to_eval_root(self):
        config = {"path": "eval_teammates/ippo/run", "actor_type": "mlp"}
        resolved = resolve_heldout_config(config, "/data/jax-aht")
        self.assertEqual(resolved["path"], str(Path("/data/jax-aht") / "eval_teammates/ippo/run"))
        self.assertEqual(config["path"], "eval_teammates/ippo/run")

    def test_absolute_path_and_scripted_entries_are_unchanged(self):
        absolute = {"path": "/elsewhere/run"}
        scripted = {"actor_type": "onion_agent", "p_onion_on_counter": 0.1}
        self.assertEqual(resolve_heldout_config(absolute, "/data/jax-aht"), absolute)
        self.assertEqual(resolve_heldout_config(scripted, "/data/jax-aht"), scripted)

    def test_missing_eval_root_keeps_relative_path(self):
        config = {"path": "eval_teammates/ippo/run"}
        self.assertEqual(resolve_heldout_config(config, None), config)


class SelectHeldoutMemberTests(unittest.TestCase):
    def setUp(self):
        self.loaded = {
            "ippo_mlp (0)": ("policy", {"w": 0}, True, None),
            "ippo_mlp (1)": ("policy", {"w": 1}, True, None),
        }

    def test_selects_the_requested_member(self):
        policy, params, label = select_heldout_member(self.loaded, "ippo_mlp", "ippo_mlp (1)")
        self.assertEqual((policy, params, label), ("policy", {"w": 1}, "ippo_mlp (1)"))

    def test_single_member_needs_no_label(self):
        single = {"ippo_mlp (0)": self.loaded["ippo_mlp (0)"]}
        self.assertEqual(select_heldout_member(single, "ippo_mlp")[2], "ippo_mlp (0)")

    def test_several_members_require_a_label(self):
        with self.assertRaises(ValueError):
            select_heldout_member(self.loaded, "ippo_mlp")

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(KeyError):
            select_heldout_member(self.loaded, "ippo_mlp", "ippo_mlp (7)")

    def test_scripted_agent_is_rejected(self):
        scripted = {"onion_agent_0.1": ("policy", None, False, None)}
        with self.assertRaises(ValueError):
            select_heldout_member(scripted, "onion_agent_0.1")


if __name__ == "__main__":
    unittest.main()
