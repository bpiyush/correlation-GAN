# Import this file in every file to setup PYTHONPATH correctly
import sys
from pathlib import Path


def get_project_root():
    """Returns project root folder."""
    return str(Path(__file__).absolute().parent.parent)


project_root = get_project_root()
sys.path.insert(0, project_root)