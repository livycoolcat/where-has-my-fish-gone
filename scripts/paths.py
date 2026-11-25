"""
One place for every path in the project.

All paths are listed in config.yaml (in the repo folder), relative to the repo,
so the scripts work the same on everyone's computer.

    from paths import path
    WEEKS_DIR = path("weeks")
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent   # the repo folder, wherever it is
CONFIG = yaml.safe_load(open(REPO / "config.yaml"))


def path(name):
    # Full path for a name in config.yaml, e.g. path("weeks")
    return REPO / CONFIG[name]


def setting(name):
    # Any other value in config.yaml, e.g. setting("splits")
    return CONFIG[name]
