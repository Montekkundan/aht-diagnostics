import unittest

from aht_diagnostics.semantics import OvercookedStateView, semantic_event


def view(
    *,
    inventory="empty",
    partner_inventory="empty",
    pots=(23,),
    position=(1, 1),
    direction=0,
):
    return OvercookedStateView(
        positions=(position, (3, 1)),
        inventories=(inventory, partner_inventory),
        pot_statuses=pots,
        directions=(direction, 0),
    )


class OvercookedSemanticTests(unittest.TestCase):
    def test_names_onion_pickup(self):
        event = semantic_event(
            view(),
            view(inventory="onion"),
            agent_index=0,
            action=5,
            reward=0.0,
        )
        self.assertEqual(event, "fetch_onion")

    def test_distinguishes_pot_placement_from_counter_drop(self):
        placed = semantic_event(
            view(inventory="onion", pots=(23,)),
            view(inventory="empty", pots=(22,)),
            agent_index=0,
            action=5,
            reward=0.0,
        )
        dropped = semantic_event(
            view(inventory="onion", pots=(23,)),
            view(inventory="empty", pots=(23,)),
            agent_index=0,
            action=5,
            reward=0.0,
        )
        self.assertEqual(placed, "place_onion_in_pot")
        self.assertEqual(dropped, "drop_onion")

    def test_names_soup_delivery(self):
        event = semantic_event(
            view(inventory="dish"),
            view(inventory="empty"),
            agent_index=0,
            action=5,
            reward=20.0,
        )
        self.assertEqual(event, "deliver_soup")

    def test_names_movement_and_blocking(self):
        moved = semantic_event(
            view(position=(1, 1)),
            view(position=(1, 2)),
            agent_index=0,
            action=1,
            reward=0.0,
        )
        blocked = semantic_event(
            view(position=(1, 1)),
            view(position=(1, 1)),
            agent_index=0,
            action=1,
            reward=0.0,
        )
        self.assertEqual(moved, "move")
        self.assertEqual(blocked, "blocked_move")

    def test_distinguishes_turning_in_place_from_a_blocked_move(self):
        turned = semantic_event(
            view(position=(1, 1), direction=0),
            view(position=(1, 1), direction=1),
            agent_index=0,
            action=1,
            reward=0.0,
        )
        blocked = semantic_event(
            view(position=(1, 1), direction=1),
            view(position=(1, 1), direction=1),
            agent_index=0,
            action=1,
            reward=0.0,
        )

        self.assertEqual(turned, "orientation_change")
        self.assertEqual(blocked, "blocked_move")


if __name__ == "__main__":
    unittest.main()
