# This script gets executed on the build machine that builds your environment,
# but before the environment is created. That means you cannot depend on system
# packages that are not installed on the build machine, so keep it simple.
# Add any dependencies required for data setup in the section below.

# /// script
# requires-python = "==3.12.*"
# dependencies = []
# ///


def main():
    # Add data download/generation/preprocessing logic here.
    pass


if __name__ == "__main__":
    main()
