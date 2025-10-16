import os
import ast
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EXCLUDE_FILES = {"__init__.py", "scan.py"}
EXCLUDE_FOLDERS = {".venv", "__pycache__", "build", "dist", ".git"}


def scan_module(file_path):
    """Return top-level classes and functions in a Python file."""
    classes, functions = [], []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", file_path, e)
    return classes, functions


def generate_init_py(folder_path, executor=None, is_root=False):
    """
    Generate __init__.py for a folder recursively.
    Returns all exported names found in this folder and its subfolders.
    """
    imports = []
    exported_names = []

    entries = [e for e in os.listdir(folder_path) if e not in EXCLUDE_FOLDERS]
    py_files = [
        e
        for e in entries
        if os.path.isfile(os.path.join(folder_path, e))
        and e.endswith(".py")
        and e not in EXCLUDE_FILES
    ]
    subfolders = [e for e in entries if os.path.isdir(os.path.join(folder_path, e))]

    # Recurse into subfolders
    for sub in subfolders:
        sub_path = os.path.join(folder_path, sub)
        sub_exports = generate_init_py(sub_path, executor=executor)
        if sub_exports:
            imports.append(f"from . import {sub}")
            exported_names.append(sub)

    # Scan Python files
    def process_file(f):
        mod_name = f[:-3]
        classes, funcs = scan_module(os.path.join(folder_path, f))
        lines = []
        names = []
        for cls in classes:
            lines.append(f"from .{mod_name} import {cls}")
            names.append(cls)
        for func in funcs:
            lines.append(f"from .{mod_name} import {func}")
            names.append(func)
        return lines, names

    if executor:
        futures = {executor.submit(process_file, f): f for f in py_files}
        for future in as_completed(futures):
            lines, names = future.result()
            imports.extend(lines)
            exported_names.extend(names)
    else:
        for f in py_files:
            lines, names = process_file(f)
            imports.extend(lines)
            exported_names.extend(names)

    # Write __init__.py
    init_path = os.path.join(folder_path, "__init__.py")
    try:
        with open(init_path, "w", encoding="utf-8") as f:
            # Top comment
            f.write(f"# Auto-generated __init__.py for folder: {folder_path}\n\n")

            # Write imports
            if imports:
                f.write("\n".join(imports) + "\n\n")

            # Write __all__ if we have any exports
            if exported_names:
                f.write("__all__ = [\n")
                for name in exported_names:
                    f.write(f"    '{name}',\n")
                f.write("]\n")
            else:
                f.write("# Empty module\n")
        logger.info("Generated %s", init_path)
    except Exception as e:
        logger.error("Failed to write %s: %s", init_path, e)

    return exported_names


def main():
    root_folder = os.path.dirname(os.path.abspath(__file__))
    logger.info("Scanning and generating __init__.py starting at %s", root_folder)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        generate_init_py(root_folder, executor=executor, is_root=True)

    logger.info("Done generating __init__.py files.")


if __name__ == "__main__":
    main()
