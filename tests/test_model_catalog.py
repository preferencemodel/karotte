"""The enumerable model catalog a backend captures at build time."""

from typing import Any

import pytest

from karotte.model_catalog import (
    CATALOG_MODEL_IDS,
    MODEL_FAMILIES,
    ModelCatalog,
    catalog,
)
from karotte.model_spec import spec_for


def test_every_id_appears_once():
    assert len(set(CATALOG_MODEL_IDS)) == len(CATALOG_MODEL_IDS)


def test_specs_match_the_ids():
    assert [spec.model for spec in catalog().models] == list(CATALOG_MODEL_IDS)


def test_specs_match_direct_resolution():
    """The catalog is a listing of `spec_for`, never a second set of rules."""
    for spec in catalog().models:
        assert spec == spec_for(spec.model)


def test_families_are_prefixes():
    for family in MODEL_FAMILIES:
        assert family.prefix.endswith("/")


def test_roundtrips_through_json():
    """A backend stores this on the build record and validates it back."""
    dumped = catalog().model_dump_json()
    assert ModelCatalog.model_validate_json(dumped) == catalog()


def test_golden_catalog(assert_matches_golden: Any):
    """Adding or dropping a model is a reviewed change, not a silent one."""
    assert_matches_golden("model_catalog.json", catalog().model_dump(mode="json"))


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-5", 128000),
        ("openai/gpt-5.6", 128000),
        ("together_ai/zai-org/GLM-5.2", 64000),
    ],
)
def test_capability_data_is_carried(model: str, expected: int):
    specs = {spec.model: spec for spec in catalog().models}
    assert specs[model].max_output_tokens == expected
