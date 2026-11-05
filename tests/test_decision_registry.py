import pytest

from decision_fixtures import build_d1, build_nanojev
from dspy_wsp.decision.families import FAMILIES, backend_class
from dspy_wsp.decision.loading import ALIASES, detect_family


@pytest.fixture(scope="module")
def d1_dir(tmp_path_factory):
    return build_d1(tmp_path_factory.mktemp("d1"))


@pytest.fixture(scope="module")
def nj_dir(tmp_path_factory):
    return build_nanojev(tmp_path_factory.mktemp("nanojev"))


def test_registry_lists_families_and_aliases():
    assert [f.name for f in FAMILIES][:2] == ["d1_omni", "nanojev"]
    assert ALIASES["d1"] == ("LiquidAI/d1-omni-600M", "02b55d7076f15129e59ab3f94783f32c4b088674")
    assert ALIASES["nanojev"] == ("C-Tianyu/NanoJev", "047b927b30882a1138fc504821b82ac145a4b81a")
    assert backend_class(FAMILIES[0]).__name__ == "D1Backend"
    assert backend_class(FAMILIES[1]).__name__ == "NanoJevBackend"


def test_detect_family_names(d1_dir, nj_dir):
    assert detect_family(str(d1_dir)) == "d1_omni"
    assert detect_family(str(nj_dir)) == "nanojev"
