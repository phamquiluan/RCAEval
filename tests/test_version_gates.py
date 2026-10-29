"""The default environment must accept any modern Python, not just the
minor versions it was last tested on."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fake the version only where RCAEval reads it: replacing sys.version_info
# globally sends numpy, torch and friends down code paths for the wrong
# interpreter (e.g. importing 3.14's annotationlib on 3.12)
FAKE = """
import collections, types
import RCAEval.utility
V = collections.namedtuple("V", "major minor micro releaselevel serial")
RCAEval.utility.sys = types.SimpleNamespace(version_info=V({major}, {minor}, 0, "final", 0))
"""


def run_as(version, code):
    major, minor = version
    return subprocess.run(
        [sys.executable, "-c", FAKE.format(major=major, minor=minor) + code],
        capture_output=True, text=True, cwd=ROOT,
    )


@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 13), (3, 15)])
def test_e2e_exposes_methods_on_any_modern_python(version):
    out = run_as(version, "import RCAEval.e2e as e; print(hasattr(e, 'baro'))")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "True"


@pytest.mark.parametrize("version", [(3, 13), (3, 15)])
def test_main_accepts_any_modern_python(version):
    out = run_as(version, "import runpy, sys; sys.argv = ['main.py', '--help']; runpy.run_path('main.py', run_name='__main__')")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Please use Python" not in out.stdout


def test_main_rejects_unsupported_python():
    out = run_as((3, 9), "import runpy, sys; sys.argv = ['main.py', '--help']; runpy.run_path('main.py', run_name='__main__')")
    assert out.returncode == 1
    assert "Please use Python 3.14" in out.stdout
