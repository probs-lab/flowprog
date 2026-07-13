#!/usr/bin/env python3
"""One-off migration utility: regenerate model_data.json's 21 reference
scenarios' `results` dicts from the current (migrated) model.

This is *not* the normal way to refresh model_data.json (see README.md --
that's driving a structural change from the calculator repo). This script is
for the elementary-exchanges migration specifically (implementation plan
section 7, migration step 2): re-baselining the reference values after a
deliberate, reviewed change to how they're computed.

Safety check: every key present in *both* the old and new results for a
scenario must match to within the same tolerance run_benchmark.verify() uses.
Only a documented set of keys is allowed to differ (new ProcessThroughput_*
keys for the boundary processes and the two bug fixes -- see README.md). Any
other difference aborts without writing, so this can't silently launder an
unrelated regression into the golden values.
"""

import json
import sys

import numpy as np

from structure import load_data, DATA_PATH
from run_benchmark import build, get_model_output, RTOL, ATOL

# Keys allowed to differ from the old reference values, with a short reason.
# Anything else differing is treated as an unreviewed regression.
ALLOWED_NEW_KEYS_PREFIXES = (
    "ProcessThroughput_SourceOf",  # 12 new boundary processes now appear as flows
    "Emissions_NGCombustion",  # renamed from Emissions_NGCombustion_NGReq* (2x
    "Emissions_NGWTT",  # double-counting fix, see ALLOWED_REMOVED_KEY_PREFIXES)
)
ALLOWED_REMOVED_KEY_PREFIXES = (
    "Emissions_NGCombustion_NGReq",  # renamed: mislabeled aggregate ("NGReq" as if
    "Emissions_NGWTT_NGReq",  # a group) folded into the double-counted total, see below
)
ALLOWED_CHANGED_KEYS_PREFIXES = (
    "Emissions_ElecReq",  # green-hydrogen quirk fix: aggregate now uses the
    # correct per-source EF instead of the ordinary EF applied to the whole total
    "Emissions_NGCombustion",  # 2x double-counting fix + rename (was
    "Emissions_NGWTT",  # Emissions_NGCombustion_NGReq / Emissions_NGWTT_NGReq)
    "EmissionsBySource_NGCombustion",  # downstream of the same 2x fix (add_sources()
    "EmissionsBySource_NGWTT",  # summed the mislabeled aggregate key in with the groups)
    "CCS",  # same root cause: the old NG loop also double-counted the
    # aggregate's own (original - abated) into the running CCS total
)


def _matches_prefix(key, prefixes):
    return any(key == p or key.startswith(p) for p in prefixes)


def main():
    data = load_data()
    _, func = build()

    new_data = json.loads(json.dumps(data))  # deep copy, cheaper than copy.deepcopy here

    n_unexpected = 0
    for name, case in data["scenarios"].items():
        old = case["results"]
        new = get_model_output(func, case["params"])

        old_keys, new_keys = set(old), set(new)
        removed = old_keys - new_keys
        added = new_keys - old_keys
        common = old_keys & new_keys

        unexpected_removed = {
            k for k in removed if not _matches_prefix(k, ALLOWED_REMOVED_KEY_PREFIXES)
        }
        unexpected_added = {
            k for k in added if not _matches_prefix(k, ALLOWED_NEW_KEYS_PREFIXES)
        }
        if unexpected_removed or unexpected_added:
            print(f"[{name}] UNEXPECTED KEY CHANGES")
            print(f"  unexpected removed: {sorted(unexpected_removed)}")
            print(f"  unexpected added:   {sorted(unexpected_added)}")
            n_unexpected += 1

        for k in common:
            if _matches_prefix(k, ALLOWED_CHANGED_KEYS_PREFIXES):
                continue
            if not np.isclose(old[k], new[k], rtol=RTOL, atol=ATOL):
                print(
                    f"[{name}] UNEXPECTED VALUE CHANGE in {k!r}: "
                    f"old={old[k]!r} new={new[k]!r}"
                )
                n_unexpected += 1

        new_data["scenarios"][name]["results"] = new

    if n_unexpected:
        print(f"\n{n_unexpected} unexpected change(s) -- NOT writing model_data.json")
        sys.exit(1)

    with open(DATA_PATH, "w") as f:
        json.dump(new_data, f, indent=2)
        f.write("\n")
    print(f"\nOK -- wrote regenerated results to {DATA_PATH}")


if __name__ == "__main__":
    main()
