from dataclasses import dataclass
from pathlib import Path
import shutil
import typing as tp
import yaml
import jax
import jinja2

import elegy
from types import ModuleType


@dataclass
class Structure:
    obj: tp.Any
    name_path: str
    members: tp.List[str]


def get(module, name_path):

    all_members = module.__all__ if hasattr(module, "__all__") else []
    all_members = sorted(all_members)

    outputs = {
        name: get(module, f"{name_path}.{name}")
        if isinstance(module, ModuleType)
        else Structure(
            obj=module,
            name_path=f"{name_path}.{name}",
            members=module.__all__ if hasattr(module, "__all__") else [],
        )
        for module, name in ((getattr(module, name), name) for name in all_members)
    }

    return {k: v for k, v in outputs.items() if v}


docs_info = get(elegy, "elegy")

# populate mkdocs
with open("mkdocs.yml", "r") as f:
    docs = yaml.safe_load(f)


[api_reference_index] = [
    index for index, section in enumerate(docs["nav"]) if "API Reference" in section
]


print(api_reference_index)

api_reference = jax.tree_map(
    lambda s: s.name_path.replace("elegy", "api").replace(".", "/") + ".md", docs_info
)

docs["nav"][api_reference_index] = {"API Reference": api_reference}

with open("mkdocs.yml", "w") as f:
    yaml.safe_dump(docs, f, default_flow_style=False, sort_keys=False)


template = """
# {{name_path}}

::: {{name_path}}
    selection:
        inherited_members: true
        members:
        {%- if members %}
        {%- for member in members %}
            - {{member}}
        {%- endfor %}
        {% else %}
            - __NONE__
        {% endif %}
"""

api_path = Path("docs/api")
shutil.rmtree(api_path, ignore_errors=True)

for structure in jax.tree_leaves(docs_info):
    filepath: Path = api_path / (
        structure.name_path.replace("elegy.", "").replace(".", "/") + ".md"
    )
    markdown = jinja2.Template(template).render(
        name_path=structure.name_path, members=structure.members
    )

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(markdown)
