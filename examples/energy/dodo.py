#!/usr/bin/env python3


from pathlib import Path


def task_extract_rdf():
    return {
        "targets": ["system-definitions/_build/probs_rdf/output.ttl"],
        "file_dep": list(Path("system-definitions").glob("*.md")),
        "actions": [
            # Clean is needed for now at least, since Sphinx extension is not
            # properly responding to things being removed and caches them too
            # aggressively.
            #
            # Also when using jupyter-cache to avoid rerunning every notebook
            # every time, the cache needs to be cleared when the rdf output
            # changes (the --all option does this).
            "jupyter-book clean --all system-definitions",
            (
                "jupyter-book build system-definitions -v --builder=custom --custom-builder=probs_rdf "
                "--config system-definitions/_config_extract_rdf.yml"
            ),
        ],
    }
