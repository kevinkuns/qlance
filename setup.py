"""
Setup
"""

import setuptools

version = '0.2.1'

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='PyTickle',
    version=version,
    url='https://github.com/kevinkuns/pytickle',
    author='Kevin Kuns',
    author_email='kkuns@mit.edu',
    description='Optomechanical, quantum noise, and control loop simulations',
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ]
)
