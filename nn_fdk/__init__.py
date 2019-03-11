# -*- coding: utf-8 -*-

"""Top-level package for Neural Network FDK algorithm."""

__author__ = """Rien Lagerwerf"""
__email__ = 'rienlagerwerf@gmail.com'


def __get_version():
    import os.path
    version_filename = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

# Import all definitions from main module.
from .nn_fdk import *
