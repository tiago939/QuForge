from setuptools import setup, find_packages

setup(
    name="quforge",
    version="0.2.0",
    author="Tiago de Souza Farias",
    author_email="tiago.farias@ufscar.br",
    description="QuForge: a library for qudit simulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tiago939/quforge",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
