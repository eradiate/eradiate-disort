import eradiate
import pytest


@pytest.fixture
def mode_mono():
    eradiate.set_mode("mono")
    yield


@pytest.fixture
def mode_ckd():
    eradiate.set_mode("ckd")
    yield
