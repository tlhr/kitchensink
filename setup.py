from setuptools import setup

setup(
    name="kitchensink",
    version="0.1",
    description="Various utilities with a focus on simulations",
    author="Thomas Löhr",
    author_email="thomas@loehr.co.uk",
    license="GPL",
    url="https://github.com/tlhr/kitchensink",
    packages=["kitchensink"],
    install_requires=[
        "numpy",
        "pandas",
        "numba",
        "sklearn",
        "mdtraj",
        "pyemma",
        "tqdm",
        "scipy",
        "networkx",
        "matplotlib",
        "h5py",
    ]
)