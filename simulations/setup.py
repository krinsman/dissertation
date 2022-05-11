import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="high_throughput_microbial_community_simulations",
    version="0.1.0",
    description="Code for simulations, and calculating some statistics, used in project",
    long_description=README,
    long_description_content_type="text/markdown",
    author="William Krinsman",
    author_email="krinsman@berkeley.edu",
    license="GPL-3",
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
)
