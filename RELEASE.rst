
=============
Release Notes
=============

This is the list of changes to sim_chime_scenario_runner between each release. For full details,
see the commit logs at http://github.com/misken/sim_chime_scenario_runner

What is it
----------

A simple Python module for working with the penn_chime model from the command line or as importable functions.

Where to get it
---------------

* Source code: http://github.com/misken/sim_chime_scenario_runner
* Documentation: See Jupyter notebooks in docs folder

sim_chime_scenario_runner 1.1.3
===============================

**Release date:** 2020-04-08

**New features**

**Improvements to existing features**

**API Changes**

Parameters and argument parsing consistent with penn-chime v1.1.3

* recovered is required but not yet implemented according to -h
* ALOS parameter names now all consistently plural "-days"

	Example file:

	    --population 5026226
	    --market-share 0.32
	    --current-hospitalized 935
	    --date-first-hospitalized 2020-02-20
	    --mitigation-date 2020-03-21
	    --current-date 2020-04-05
	    --recovered 0
	    --relative-contact-rate 0.30
	    --hospitalized-rate 0.025
	    --icu-rate 0.0075
	    --ventilated-rate 0.005
	    --infectious-days 14
	    --hospitalized-days 7
	    --icu-days 9
	    --ventilated-days 10
	    --n-days 120

**Bug Fixes**


sim_chime_scenario_runner 1.1.2
===============================

**Release date:** 2020-04-07

**New features**

:mod:`sim_chime_scenario_runner` consists of the following things and features

It is a simple Python module for working with the penn_chime model from the command line or as importable functions.

* works with penn-chime v1.1.2
* assumes that you've pip installed `penn_chime` either per https://github.com/CodeForPhilly/chime/pull/249 from a local clone of the chime repo or from pypi if it's eventually put up there
* [OPTIONAL] You can do a `pip install .` from the directory containing setup.py if you want to install into a virtual environment
* allows running simulations from command line (like cli.py in penn_chime)
* is importable so can also run simulations via function call
* includes a few additional command line (or passable) arguments, including:
  - standard CHIME input config filename is a required input
  - a scenario name (prepended to output filenames)
  - output path
* after a simulation scenario is run, a results dictionary is created that contains:
  - the scenario name
  - the standard admits, census, and sim_sir_w_date dataframes
  - the dispositions dataframe
  - a dictionary containing the input parameters
  - a dictionary containing important intermediate variable values such as beta, doubling_time, ...
* writes out the results 
  - dataframes to csv
  - dictionaries to json
* (WIP) runs multiple scenarios corresponding to user specified ranges for one or more input variables.

I did my best to make my code easy to maintain if CHIME changes.

- created arg parser just for my added input parameters
- uses standard CHIME input config file
- calls penn_chime's argument parser to parse the input config file
- the main simulation function signature is `sim_chime(scenario: str, p: Parameters):`
- `sim_chime()` returns a tuple containing the model object and the results dictionary described above.

**Improvements to existing features**

**API Changes**

**Bug Fixes**







hillmaker 0.1.0
===============

**Release date:** 2016-01-22

**New features**


