# Script to show who injects what and when and where into the PYTHONPATH
# ChatGPT 2025-09-01

import os, sys, site, pathlib

def explain_sys_path():
    print("PYTHONPATH =", os.environ.get("PYTHONPATH", ""))
    print("\n--- sys.path analysis ---")

    site_dirs = set(site.getsitepackages() + [site.getusersitepackages()])
    cwd = pathlib.Path().resolve()

    for p in sys.path:
        origin = []
        path = pathlib.Path(p).resolve() if p else cwd

        # Empty string "" means current working dir
        if p == "":
            origin.append("current working dir")

        # From PYTHONPATH
        if os.environ.get("PYTHONPATH"):
            for envdir in os.environ["PYTHONPATH"].split(os.pathsep):
                if path == pathlib.Path(envdir).resolve():
                    origin.append("from $PYTHONPATH")

        # Inside site-packages
        for sdir in site_dirs:
            if str(path).startswith(str(pathlib.Path(sdir).resolve())):
                origin.append("site-packages")

        # Egg-link files (editable installs)
        egglinks = list(pathlib.Path(s).glob("*.egg-link") for s in site_dirs)
        for links in egglinks:
            for link in links:
                try:
                    target = pathlib.Path(link.read_text().splitlines()[0]).resolve()
                    if path == target:
                        origin.append(f"editable install via {link.name}")
                except Exception:
                    pass

        if not origin:
            origin = ["(unknown, maybe VS Code .env or injected)"]

        print(f"{p!r} -> {', '.join(origin)}")

explain_sys_path()
