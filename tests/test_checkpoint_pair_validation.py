from scripts.validate_checkpoint_pairs import PARTNER_SPECS


def test_every_checkpoint_has_an_explicit_training_partner() -> None:
    assert set(PARTNER_SPECS) == {
        "independent_agent_0",
        "independent_agent_0.4",
        "onion_agent_0.1",
        "plate_agent_0.1",
    }
    assert PARTNER_SPECS["onion_agent_0.1"]["role"] == "onion"
    assert PARTNER_SPECS["plate_agent_0.1"]["role"] == "plate"
