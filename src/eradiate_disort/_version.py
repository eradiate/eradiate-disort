# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

from importlib.metadata import PackageNotFoundError, version

try:
    _version = version("eradiate-disort")
except PackageNotFoundError as e:
    raise PackageNotFoundError(
        "eradiate-disort is not installed; please install it in your "
        "Python environment."
    ) from e
