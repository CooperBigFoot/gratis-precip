from setuptools import setup, find_packages

setup(
    name="gratis_precip",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
