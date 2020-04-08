## sim_chime_scenario_runner

**[2020-04-08] TEMPORARILY OUT OF COMMISSION - Broken due to CLI changes in penn_chime - fixing asap**

A simple Python module for working with the penn_chime model from the command line or as importable functions. 

Has its own repo now: https://github.com/misken/sim_chime_scenario_runner

* A Jupyter notebook demo showing its use: [using_sim_chime_scenario_runner.ipynb](https://github.com/misken/sim_chime_scenario_runner/blob/master/demos/using_sim_chime_scenario_runner.ipynb)

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

I borrowed some code from the CHIME project (https://github.com/CodeForPhilly/chime) but did my best
to make my code easy to maintain if CHIME changes.

- created arg parser just for my added input parameters
- uses standard CHIME input config file
- call `penn_chime.cli.parse_args()` to parse the input config file
- the main simulation function signature is `sim_chime(scenario: str, p: Parameters):`
- `sim_chime()` returns a tuple containing the model object and the results dictionary described above.


