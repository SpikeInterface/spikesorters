from setuptools import setup, find_packages

d = {}
exec(open("spikesorters/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "spikesorters"

setup(
    name=pkg_name,
    version=version,
    author="Alessio Buccino, Cole Hurwitz, Samuel Garcia, Jeremy Magland, Matthias Hennig",
    author_email="alessiop.buccino@gmail.com",
    description="Python wrappers for popular spike sorters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpikeInterface/spikesorters",
    packages=find_packages(),
    package_data={},
    include_package_data=True,
    install_requires=[
        'numpy',
        'spikeextractors',
        'spiketoolkit',
        'requests'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
