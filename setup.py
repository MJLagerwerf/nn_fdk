#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import os.path

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

with open(os.path.join('nn_fdk','VERSION')) as version_file:
    version = version_file.read().strip()

requirements = [
	'ddf_fdk',
	'numexpr',
	'h5py',
	'sacred',
	'tqdm',
    'pymongo'
    # Add your project's requirements here, e.g.,
    # 'astra-toolbox',
    # 'sacred>=0.7.2',
    # 'tables==3.4.4',
]

setup_requirements = [ ]

test_requirements = [ ]

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'recommonmark',
    # Other
    'watchdog',
    'coverage',
    
    ]

setup(
    author="Rien Lagerwerf",
    author_email='rienlagerwerf@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Support code for the Neural Network FDK algorithm paper",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='nn_fdk',
    name='nn_fdk',
    packages=find_packages(include=['nn_fdk']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={ 'dev': dev_requirements },
    url='https://github.com/mjlagerwerf/nn_fdk',
    version=version,
    zip_safe=False,
)
