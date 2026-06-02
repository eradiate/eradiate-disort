# Align the macOS wheel platform tag with the interpreter's deployment target.
#
# scikit-build-core tags locally-built wheels via `packaging.tags`, which uses
# the *running* macOS version (`platform.mac_ver()`). uv, however, computes wheel
# compatibility from the interpreter's *build-time* target
# (`sysconfig MACOSX_DEPLOYMENT_TARGET`). On a recent macOS these disagree (e.g.
# host 26.x vs interpreter 11.0), so uv rejects the wheel it just built.
#
# Pinning MACOSX_DEPLOYMENT_TARGET to the interpreter's own sysconfig value makes
# the produced tag equal to uv's acceptance ceiling by construction, on every
# platform and across future conda-forge baseline bumps. No-op off macOS.
if [ "$(uname -s)" = "Darwin" ]; then
    _macos_target="$(python -c 'import sysconfig; print(sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET") or "")')"
    if [ -n "${_macos_target}" ]; then
        export MACOSX_DEPLOYMENT_TARGET="${_macos_target}"
    fi
    unset _macos_target
fi
