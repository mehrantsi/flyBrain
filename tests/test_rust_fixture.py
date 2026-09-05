from __future__ import annotations

from tools.generate_rust_fixture import DEFAULT_PATH, load_fixture, make_fixture, validate_fixture


def test_committed_rust_fixture_regenerates_and_validates() -> None:
    committed = load_fixture(DEFAULT_PATH)

    assert committed == make_fixture()
    validate_fixture(committed)
