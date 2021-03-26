import setuptools
import os.path
import re



with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join("EXOSIMS","__init__.py"), "r") as f:
    version_file = f.read()

version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",\
        version_file, re.M)

if version_match:
    version_string = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="exo-det-box",
    version=version_string,
    author="Dean Robert Keithly",
    author_email="drk94@cornell.edu",
    description="Methods for finding planets sharing a common (s,dmag)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SIOSlab/exo-det-box",
    packages=setuptools.find_packages(exclude=['tests*']),
    include_package_data=True,
    install_requires=['numpy','scipy','astropy','EXOSIMS','ortools'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    ext_modules = extensions
)
