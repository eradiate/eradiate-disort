from eradiate_disort.testing.fixtures import *  # noqa: F403, I001
from eradiate_disort.testing.plotting import ert_plt  # noqa: F401


def pytest_addoption(parser):
    parser.addoption(
        "--plots",
        nargs="?",
        const="plots",
        default=None,
        metavar="DIR",
        help="Save diagnostic plots to DIR (default: plots/).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "plotting: test generates diagnostic plots (skipped implicitly when --plots is absent)",
    )
