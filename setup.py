import sys

from setuptools import setup, find_packages

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Spinning Up repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

setup(
    name='spinup',
    version=0.1,
    packages=find_packages(),
    install_requires=[
        'cloudpickle',
        'gym[atari,box2d,classic_control]',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'Pillow',
        'pytest',
        'psutil',
        'scipy',
        'seaborn',
        'torch',
        'torchvision',
        'torch_vqvae',
        'tqdm'
    ],
    description="Teaching tools for introducing people to deep RL.",
    author="Joshua Achiam",
)
