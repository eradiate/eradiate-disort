import eradiate
import pytest


@pytest.fixture
def mode_mono():
    eradiate.set_mode("mono")
    yield
