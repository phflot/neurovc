import os
from setuptools import setup, find_packages

try:
    import tomllib
except ImportError:
    import tomli as tomllib

pyproject_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project_data = pyproject_data.get("project", {})
dependencies = project_data.get("dependencies", [])

setup(
    name=project_data.get("name", "unknown"),
    version=project_data.get("version", "0.0.0"),
    description=project_data.get("description", ""),
    author=(project_data.get("authors", [{}])[0].get("name") or ""),
    author_email=(project_data.get("authors", [{}])[0].get("email") or ""),
    license=project_data.get("license", {}).get("text", ""),
    packages=find_packages(),
    install_requires=dependencies,
    python_requires=">=3.6",
)
