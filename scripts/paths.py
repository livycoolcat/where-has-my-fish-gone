"""
One place for every path in the project.

All paths are listed in config.yaml (in the repo folder), relative to the repo,
so the scripts work the same on everyone's computer.

If your data is laid out differently (e.g. on OneDrive), create config.local.yaml
next to config.yaml with only the paths you need to change. It is not committed,
and its values replace the ones in config.yaml. See config.local.example.yaml.

    from paths import path
    WEEKS_DIR = path("weeks")
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent   # the repo folder, wherever it is
CONFIG = yaml.safe_load(open(REPO / "config.yaml"))

# Personal overrides (not committed)
LOCAL = REPO / "config.local.yaml"
if LOCAL.exists():
    CONFIG.update(yaml.safe_load(open(LOCAL)) or {})


def path(name):
    # Full path for a name in config.yaml, e.g. path("weeks").
    # Relative paths are inside the repo; full paths (e.g. C:/Users/... or /Users/...) are used as they are.
    return REPO / Path(CONFIG[name]).expanduser()


def setting(name):
    # Any other value in config.yaml, e.g. setting("splits")
    return CONFIG[name]
