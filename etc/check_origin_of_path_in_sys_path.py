
# For each path in sys.path, check where path in sys.path comes from
# ChatGPT 2026-09-01

import sys, site, os, pathlib, inspect

print("=== PYTHONPATH ===")
print(os.environ.get("PYTHONPATH", "(not set)"))

print("\n=== sys.path (with origins) ===")
for p in sys.path:
    origin = None

    # Check if path comes from site-packages .pth files
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        pth_dir = pathlib.Path(sp)
        if pth_dir.exists():
            for f in pth_dir.glob("*.pth"):
                try:
                    lines = f.read_text().splitlines()
                    if any(p in line for line in lines):
                        origin = f"{f} (.pth)"
                        break
                except Exception:
                    pass

    # Check for egg-link
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        egg_link = pathlib.Path(sp) / "flopy.egg-link"
        if egg_link.exists() and str(p).startswith(str(egg_link.parent)):
            origin = f"{egg_link} (egg-link)"

    # Check for sitecustomize or usercustomize
    if "sitecustomize" in sys.modules:
        mod = sys.modules["sitecustomize"]
        origin = f"sitecustomize.py -> {inspect.getfile(mod)}"
    if "usercustomize" in sys.modules:
        mod = sys.modules["usercustomize"]
        origin = f"usercustomize.py -> {inspect.getfile(mod)}"

    print(f"{p!r} -> {origin or 'builtin/unknown'}")



