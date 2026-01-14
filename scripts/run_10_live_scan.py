"""Deprecated wrapper.

Use scripts/run_full_scan.py instead. This file delegates to the new script
to maintain backward compatibility with existing tooling.
"""

import os
import sys

# Ensure project root is on sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_full_scan import main

if __name__ == "__main__":
    main()
