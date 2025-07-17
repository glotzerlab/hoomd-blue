# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Generate one .rst file for each module and class in the HOOMD-blue public API.

``generate-toctree`` automatically generates the Sphinx .rst files for the HOOMD-blue
API documentation. The public API is defined by the items in the `__all__` list in
each package/module for compatibility with ruff's identification of exported APIs.
"""

import sys
import inspect
import os
from pathlib import Path

TOPLEVEL = ["hpmc", "mpcd", "md"]
exit_value = 0


def generate_member_rst(path, full_module_name, name, type):
    """Generate the rst file that describes a class or data element.

    Does not overwrite existing files. This allows files to be automatically created
    and then customized as needed. 80+% of the files should not need customization.
    """
    # Generate the file {name}.rst
    underline = "=" * len(name)

    member_rst = f"{name}\n{underline}\n\n"
    member_rst += f".. py:currentmodule:: {full_module_name}\n\n"

    member_rst += f".. auto{type}:: {name}\n"

    # set :members: and :show-inheritance: for classes. Developers can remove these
    # individually when they are not appropriate. Unfortunately, we cannot make these
    # default in `conf.py` because there is no way to opt out of the default.
    if type == "class":
        member_rst += "   :members:\n"
        member_rst += "   :show-inheritance:\n"

    destination = (path / name.lower()).with_suffix(".rst")

    if destination.exists():
        return

    destination.write_text(member_rst)


def generate_module_rst(path, module):
    """Generate the rst file that describes a module.

    Always overwrites the file to automatically update the table of contents when
    adding new classes.
    """
    global exit_value

    full_module_name = module.__name__
    module_name = full_module_name.split(".")[-1]

    # Alphabetize the items
    module_all = getattr(module, "__all__", None)
    if module_all is None:
        exit_value = 1
        print(f"Warning: {full_module_name} is missing __all__")
        return

    sorted_all = module_all.copy()
    sorted_all.sort()

    if len(sorted_all) > 0:
        path.mkdir(exist_ok=True)

    # Split the items into modules and class members
    submodules = []
    classes = []
    functions = []

    for member_name in sorted_all:
        member = getattr(module, member_name)
        if inspect.ismodule(member):
            submodules.append(member_name)
            generate_module_rst(path / member_name, member)

        if inspect.isclass(member):
            classes.append(member_name)
            generate_member_rst(path, full_module_name, member_name, "class")

        if inspect.isfunction(member):
            functions.append(member_name)
            generate_member_rst(path, full_module_name, member_name, "function")

        # data members should be documented directly in the module's docstring, and
        # are ignored here.

    # Generate the file module-{module_name}.rst
    module_underline = "=" * len(module_name)

    module_rst = f"{module_name}\n{module_underline}\n\n"
    module_rst += f".. automodule:: {full_module_name}\n"
    module_rst += "   :members:\n"
    if len(classes) > 0 or len(functions) > 0:
        module_rst += f"   :exclude-members: {','.join(classes + functions)}\n\n"

    if len(submodules) > 0:
        module_rst += ".. rubric:: Modules\n\n.. toctree::\n    :maxdepth: 1\n\n"
        for submodule in submodules:
            if submodule not in TOPLEVEL:
                module_rst += f"    {module_name}/module-{submodule}\n"
        module_rst += "\n"

    if len(classes) > 0:
        module_rst += ".. rubric:: Classes\n\n.. toctree::\n    :maxdepth: 1\n\n"
        for class_name in classes:
            module_rst += f"    {module_name}/{class_name.lower()}\n"
        module_rst += "\n"

    if len(functions) > 0:
        module_rst += ".. rubric:: Functions\n\n.. toctree::\n    :maxdepth: 1\n\n"
        for function_name in functions:
            module_rst += f"    {module_name}/{function_name.lower()}\n"
        module_rst += "\n"

    # ensure there is only one newline at the end of the file
    module_rst = module_rst.rstrip()
    module_rst += "\n"
    file = (path.parent / ("module-" + module_name)).with_suffix(".rst")
    file.write_text(module_rst)


if __name__ == "__main__":
    doc_dir = Path(__file__).parent
    repository_dir = doc_dir.parent
    sys.path.insert(0, str(repository_dir))

    os.environ["SPHINX"] = "1"

    import hoomd

    generate_module_rst(doc_dir / "hoomd", hoomd)
    sys.exit(exit_value)
