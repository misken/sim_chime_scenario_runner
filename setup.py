"""Setup file for chime scenario runner
"""
__version__ = "1.1.2"
__author__ = "misken"  

from os import path
from setuptools import setup, find_packages, find_namespace_packages


setup(
    name="sim_chime_scenario_runner",
    version=__version__,
    author=__author__,
    author_email="",
    description="Wrapper for penn_chime to facilitate scenario running/management",
    url="https://github.com/misken/c19",
    packages=['sim_chime_scenario_runner'],
    install_requires=[
        "numpy",
        "pandas",
        "penn-chime",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    entry_points = {
        'console_scripts': ['sim_chime_scenario_runner=sim_chime_scenario_runner.sim_chime_scenario_runner:main'],
    },
    keywords=[],
    include_package_data=True,
)

