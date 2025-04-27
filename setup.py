from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="downscalepy",
    version="0.1.0",
    author="Ana Luisa Barbosa",
    author_email="anabrs1@gmail.com",
    description="Python package for downscaling land-use change projections",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anabrs1/downscalepy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
