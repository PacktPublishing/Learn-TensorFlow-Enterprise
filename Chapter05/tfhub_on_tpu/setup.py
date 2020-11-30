from setuptools import find_packages
from setuptools import setup

setup(
    name='official',
    install_requires=['IPython', 'keras-tuner', 'tensorflow-datasets~=3.1', 'tensorflow_hub>=0.6.0'],
    packages=find_packages()
)
