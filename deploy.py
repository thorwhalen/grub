"""Manual build-and-upload helper. Importing this module has no side effects.

Run it explicitly (``python deploy.py``) to build and upload to PyPI. CI (wads)
normally publishes releases, so this is only a manual fallback.
"""

import subprocess
import sys

BUILD_COMMAND = (sys.executable, "setup.py", "sdist", "bdist_wheel")
UPLOAD_COMMAND = (sys.executable, "-m", "twine", "upload", "dist/*")


def main():
    # TODO: Both of these are python. Change to use python objects directly.
    print("--------------------------- setup_output ---------------------------")
    subprocess.run(BUILD_COMMAND, check=True)

    print("--------------------------- upload_output ---------------------------")
    subprocess.run(UPLOAD_COMMAND, check=True)


if __name__ == "__main__":
    main()
