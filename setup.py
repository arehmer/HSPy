from setuptools import setup, find_packages

with open("config/requirements.txt") as requirement_file:
    requirements = [line.rstrip() for line in requirement_file]

setup(
    name="HSPy",
    description="Python Interface for Heimann Sensor Thermopile Arrays.",
    version="0.0.1",
    author="Alexander Rehmer",
    author_email="rehmer@heimannsensor.com",
    install_requires=requirements,
    packages=find_packages(), # package = any folder with an __init__.py file
    package_data={
        # "" means apply to all packages; adjust the key to a specific
        # package name if you only want it in one folder, e.g. "HSPy.data"
        "hspy.arraytypes": ["*.json"],
    },
    include_package_data=True
)