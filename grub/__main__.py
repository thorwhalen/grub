"""Run grub as a module: ``python -m grub SOURCE [QUERY ...]``."""

import sys

from grub.cli import main

if __name__ == "__main__":
    sys.exit(main())
