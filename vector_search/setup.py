from setuptools import setup, find_packages

setup(
    name="vector_search",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)