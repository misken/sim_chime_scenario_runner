## sim_chime_scenario_runner

A simple Python module for working with the penn_chime model from the command line or as importable functions. 

### sim_chime_scenario_runner 2.1.0-chime.1.1.3

Added dynamic relative contact rates. Needs validation work but idea seems reasonable.

See [dynamic_rcr_runner.ipynb](https://github.com/misken/sim_chime_scenario_runner/blob/master/demos/dynamic_rcr_runner.ipynb) and [release notes](RELEASE.md) for details.

### sim_chime_scenario_runner 2.0.0-chime.1.1.3

Added a bunch of new functionality but still more to do, especially for experiments (batches of scenarios).
See [release notes](RELEASE.md) for details.

### About using

* A Jupyter notebook demo showing its use: [using_sim_chime_scenario_runner.ipynb](https://github.com/misken/sim_chime_scenario_runner/blob/master/demos/using_sim_chime_scenario_runner.ipynb)

* assumes that you've pip installed `penn_chime` per [these instructions](https://github.com/misken/c19/blob/master/penn_chime_cli_quickstart.md)
* [OPTIONAL] You can do a `pip install .` of `sim_chime_scenario_runner` from the directory containing setup.py if you want to install into a virtual environment
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
- calls penn_chime's argument parser to parse the input config file
- the main simulation function signature is `sim_chime(scenario: str, p: Parameters):`
- `sim_chime()` returns a tuple containing the model object and the results dictionary described above.


