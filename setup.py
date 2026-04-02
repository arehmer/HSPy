from setuptools import setup, find_packages

with open("config/requirements.txt") as requirement_file:
    requirements = [line.rstrip() for line in requirement_file]

setup(
    name="HSPy",
    description="A simple package.",
    version="1.0.0",
    author="Name",
    author_email="Name@domain.com",
    install_requires=requirements,
    packages=find_packages(), # package = any folder with an __init__.py file
)