from scripts.summarize_diagnostic_suite import summary_metadata


def test_confirmation_summary_metadata_does_not_claim_exposed_development() -> None:
    metadata = summary_metadata(
        {"status": "prospective_diagnostic_baseline_confirmation"},
        {"status": "prospective_diagnostic_baseline_confirmation_capable_control"},
    )

    assert metadata["status"] == "prospective_diagnostic_suite_summary"
    assert "prospective confirmation split" in metadata["claim_scope"]


def test_development_summary_metadata_retains_development_boundary() -> None:
    metadata = summary_metadata(
        {"status": "development_result"},
        {"status": "development_result"},
    )

    assert metadata["status"] == "derived_diagnostic_suite_summary"
    assert "not independent confirmation" in metadata["claim_scope"]
