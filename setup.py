# setup.py
from setuptools import setup, find_packages

setup(
    name='SslabLU',
    version='1.0',
    packages=['multislab','matAssembly','solver'],
    license="MIT",
    author="Simon Dirckx",
    author_email='simon.dirckx@austin.utexas.edu',
    url='https://github.com/SimonDirckx/SslabLU',
    install_requires=[
        'numpy','matplotlib','scipy', 'jax', 'pytest'
    ],
)
