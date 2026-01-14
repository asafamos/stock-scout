"""Deprecated wrapper.

Use scripts/run_full_scan.py instead. This file delegates to the new script
to maintain backward compatibility with existing tooling.
"""

from scripts.run_full_scan import main

if __name__ == "__main__":
    main()
