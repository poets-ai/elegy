import re
from pathlib import Path

import typer


# NOTE: this script could be written bash using sed, but I'm not sure if it's worth it
def main(release_name: str):
    release_name = release_name.replace("-create-release", "")

    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    pyproject_text = pyproject_path.read_text()
    pyproject_text = re.sub(
        r'version = ".*"',
        f'version = "{release_name}"',
        pyproject_text,
        count=1,
    )
    pyproject_path.write_text(pyproject_text)

    # Update __init__.py
    init_path = Path("elegy", "__init__.py")
    init_text = init_path.read_text()
    init_text = re.sub(
        r'__version__ = "(.*?)"',
        f'__version__ = "{release_name}"',
        init_text,
        count=1,
    )
    init_path.write_text(init_text)


if __name__ == "__main__":
    typer.run(main)
