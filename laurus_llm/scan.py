import os
import ast
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------
# Configuration
# -------------------------
EXCLUDE_FILES = {"__init__.py", "scan.py"}
EXCLUDE_FOLDERS = {".venv", "__pycache__", "build", "dist", ".git"}


# -------------------------
# AST scanning
# -------------------------
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


# -------------------------
# Generate __init__.py
# -------------------------
def generate_init_py(folder_path, executor=None, is_root=False):
    """Generate __init__.py for a folder recursively."""
    imports = []
    tasks = []

    entries = [e for e in os.listdir(folder_path) if e not in EXCLUDE_FOLDERS]
    py_files = [
        e
        for e in entries
        if os.path.isfile(os.path.join(folder_path, e))
        and e.endswith(".py")
        and e not in EXCLUDE_FILES
    ]
    subfolders = [e for e in entries if os.path.isdir(os.path.join(folder_path, e))]

    # Recurse into subfolders first
    for sub in subfolders:
        generate_init_py(os.path.join(folder_path, sub), executor=executor)

    # Scan Python files (possibly threaded)
    if executor:
        futures = {
            executor.submit(scan_module, os.path.join(folder_path, f)): f
            for f in py_files
        }
        for future in as_completed(futures):
            f = futures[future]
            mod_name = f[:-3]
            classes, funcs = future.result()
            for cls in classes:
                imports.append(f"from .{mod_name} import {cls}")
            for func in funcs:
                imports.append(f"from .{mod_name} import {func}")
    else:
        for f in py_files:
            mod_name = f[:-3]
            classes, funcs = scan_module(os.path.join(folder_path, f))
            for cls in classes:
                imports.append(f"from .{mod_name} import {cls}")
            for func in funcs:
                imports.append(f"from .{mod_name} import {func}")

    # Write __init__.py only if there are imports or if it's root
    init_path = os.path.join(folder_path, "__init__.py")
    try:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated __init__.py\n")
            if imports:
                f.write("\n".join(imports) + "\n")
            elif is_root:
                f.write("# Root package init (empty for now)\n")
        logger.info("Generated %s", init_path)
    except Exception as e:
        logger.error("Failed to write %s: %s", init_path, e)


# -------------------------
# Entry point
# -------------------------
def main():
    root_folder = os.path.dirname(os.path.abspath(__file__))
    logger.info("Scanning and generating __init__.py starting at %s", root_folder)

    # Use threads for faster scanning
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        # Generate subfolder __init__.py first
        generate_init_py(root_folder, executor=executor, is_root=False)

    # Generate root __init__.py last
    generate_init_py(root_folder, is_root=True)

    logger.info("Done generating __init__.py files.")


if __name__ == "__main__":
    main()
