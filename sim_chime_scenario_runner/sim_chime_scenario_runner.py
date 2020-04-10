"""
A combined CLI and callable version of the CHIME simulation model.

We adapted the CLI application in the CHIME project (https://github.com/CodeForPhilly/chime).

- added scenario and output_path parameters (separate from main Parameters object)
- added ability to use an input file, command line args, or DEFAULTS to instantiate model
- added ability to import this and call sim_chime() function from external apps so that we can run tons of scenarios over ranges of input parameters
- output is a dictionary of the standard CHIME dataframes as well as dictionaries
containing parameter and variable values.
- also writes out csvs
- it's in very early stages.

"""

import os
from collections import OrderedDict
from pathlib import Path

from argparse import (
    Action,
    ArgumentParser,
)
from datetime import datetime
from typing import Dict, List

from pandas import DataFrame
import numpy as np
import pandas as pd

from penn_chime.constants import CHANGE_DATE
from penn_chime.parameters import Parameters, Disposition
from penn_chime.models import SimSirModel as Model
import penn_chime.models as models
import penn_chime.cli as cli

import sys
import json


def parse_args():
    """Parse args."""
    parser = ArgumentParser(description="SEMI-CHIME")
    parser.add_argument("parameters", type=str, help="CHIME config (cfg) file")
    parser.add_argument(
        "--scenario", type=str, default=datetime.now().strftime("%Y.%m.%d.%H.%M."),
        help="Prepended to output filenames. (default is current datetime)"
    )
    parser.add_argument(
        "--output-path", type=str, default="", help="location for output file writing",
    )
    parser.add_argument(
        "--market-share", type=str, default=None, help="csv file containing date and market share (<=1.0)",
    )
    parser.add_argument(
        "--stack", action='store_true', help="Include to stack market share and scenario wide/long dfs",
    )
    parser.add_argument(
        "--actual", type=str, default=None,
        help="csv file containing day, date and actual measures (e.g. admits and census",
    )

    parser.add_argument(
        "--experiment", type=str,
        help="Undocumented feature: if not None, sim_chimes() function is called and a whole bunch of scenarios are run."
    )
    parser.add_argument("--quiet", action='store_true',
                        help="If True, suppresses output messages (default=False")

    return parser.parse_args()


def create_params_from_file(file):
    """
    Create CHIME Parameters object from input config file

    :param file:
    :return:
    """

    args = ['--parameters', file]
    p = Parameters.create(os.environ, args)

    return p



def sim_chime(scenario: str, p: Parameters):
    """
    Run one chime simulation.

    :param scenario:
    :param p:
    :return: Tuple (model, results dictionary)
    """

    input_params_dict = vars(p)

    # Run the model
    m = Model(p)

    # Gather results
    results = gather_sim_results(m, scenario, input_params_dict)
    return m, results


def write_results(results, scenario, path):
    """

    :param results:
    :param scenario:
    :param path:
    :return:
    """

    # Results dataframes
    for df, name in (
        (results["sim_sir_w_date_df"], "sim_sir_w_date"),
        (results["dispositions_df"], "dispositions"),
        (results["admits_df"], "admits"),
        (results["census_df"], "census"),
        (results["adm_cen_wide_df"], "adm_cen_wide"),
        (results["adm_cen_long_df"], "adm_cen_long"),
    ):
        df.to_csv(path + scenario + '_' + name + ".csv", index=True)

    # Variable dictionaries
    with open(path + scenario + "_inputs.json", "w") as f:
        # # Convert date to string to make json happy
        # # “%Y-%m-%d”
        # results['input_params_dict']['date_first_hospitalized'] = results['input_params_dict'][
        #     'date_first_hospitalized'].strftime("%Y-%m-%d")

        json.dump(results['input_params_dict'], f, default=str)

    with open(path + scenario + "_key_vars.json", "w") as f:
        json.dump(results['important_variables_dict'], f)


def write_experiment_results(cons_dfs, param_dict_list, scenarios, path):
    """

    :param admits_df:
    :param census_df:
    :param param_vars_df:
    :param scenarios:
    :param path:
    :return:
    """

    # Results dataframes
    for df_name, df in cons_dfs.items():
        df.to_csv(path + scenarios + '_' + df_name + ".csv", index=True)

    # Input dictionaries
    # Variable dictionaries
    with open(path + scenarios + "_inputs.json", "w") as f:
        json.dump(param_dict_list, f, default=str)


def gather_sim_results(m, scenario, input_params_dict):
    """

    :param m:
    :param scenario:
    :param input_params_dict:
    :return:
    """

    # Get key input/output variables
    intrinsic_growth_rate = m.intrinsic_growth_rate
    gamma = m.gamma     # Recovery rate
    beta = m.beta       # Contact rate

    # r_t is r_0 after distancing
    r_t = m.r_t
    r_naught = m.r_naught
    doubling_time_t = m.doubling_time_t

    intermediate_variables = OrderedDict({
        'result_type': 'simsir',
        'scenario': scenario,
        'intrinsic_growth_rate': intrinsic_growth_rate,
        'gamma': gamma,
        'beta': beta,
        'r_naught': r_naught,
        'r_t': r_t,
        'doubling_time_t': doubling_time_t,
    })

    wide_df, long_df = join_and_melt(m.admits_df, m.census_df, scenario)

    results = {
        'result_type': 'simsir',
        'scenario': scenario,
        'input_params_dict': input_params_dict,
        'important_variables_dict': intermediate_variables,
        'sim_sir_w_date_df': m.sim_sir_w_date_df,
        'dispositions_df': m.dispositions_df,
        'admits_df': m.admits_df,
        'census_df': m.census_df,
        'adm_cen_wide_df': wide_df,
        'adm_cen_long_df': long_df
    }
    return results

def join_and_melt(adm_df, cen_def, scenario):
    """
    Create wide and long DataFrames with combined admit and census data suitable for
    plotting with Seaborn or ggplot2 (in R).

    :param adm_df:
    :param cen_def:
    :param scenario:
    :return:
    """

    wide_df = pd.merge(adm_df, cen_def, left_index=True, right_index=True,
                       suffixes=('_adm', '_cen'), validate="1:1")

    wide_df['scenario'] = scenario

    pd.testing.assert_series_equal(wide_df['day_adm'],
                                         wide_df['day_cen'], check_names=False)

    pd.testing.assert_series_equal(wide_df['date_adm'],
                                         wide_df['date_cen'], check_names=False)

    wide_df.rename({'day_adm': 'day', 'date_adm': 'date'}, axis='columns', inplace=True)

    wide_df.drop(['day_cen', 'date_cen'], axis='columns', inplace=True)

    long_df = pd.melt(wide_df,
                      id_vars=['scenario', 'day', 'date'],
                      var_name='dispo_measure', value_name='cases')

    return wide_df, long_df


def sim_chimes(experiment: str, p: Parameters):
    """
    Run many chime simulations - demo.

    Need to decide on argument passing

    :param experiment:
    :param params:

    :return:
    """

    base_input_params_dict = vars(p)

    which_param_set = ''
    # Check which of date_first_hospitalized and doubling_time is set
    which_param_set = ''
    if p.date_first_hospitalized is not None and p.doubling_time is None:
        which_param_set = 'date_first_hospitalized'
    elif p.date_first_hospitalized is None and p.doubling_time is not None:
        which_param_set = 'doubling_time'
    else:
        print("Gonna be trouble. Either date_first_hospitalized or doubling_time should be set.")

    # Create a range of social distances
    relative_contact_rates = np.arange(0.05, 0.80, 0.05)

    # Create range of mitigation dates
    dates = pd.date_range('2020-03-21', '2020-03-27').to_pydatetime()
    mitigation_dates =[d.date() for d in dates]

    # Create a range of recovery times
    infectious_days = np.arange(7.00, 14.0, 1.00)

    num_scenarios = len(relative_contact_rates)

    # We can store outputs any way we want. For this demo, just going to
    # use a master list. # This will be a list of dicts of the
    # result dataframes (+ 1 dict containing the scenario inputs)

    results_list = []
    scenarios_list = [(md, rc, id) for md in mitigation_dates
                      for rc in relative_contact_rates
                      for id in infectious_days]

    for (mit_date, rcr_pct, inf_days) in scenarios_list:

        sim_scenario = '{}_md{:%m%d}rc{:.0f}id{}'.format(experiment, mit_date, 100 * rcr_pct, int(inf_days))

        # Update the parameters for this scenario
        p.mitigation_date = mit_date
        p.relative_contact_rate = rcr_pct
        p.infectious_days = inf_days

        if which_param_set == 'date_first_hospitalized':
            p.doubling_time = None
        else:
            p.date_first_hospitalized = None

        input_params_dict = OrderedDict(vars(p))

        # Run the model
        m = Model(p)

        # Gather results
        results = gather_sim_results(m, sim_scenario, input_params_dict)

        # Append results to results list
        results_list.append(results.copy())

    return results_list


def consolidate_scenarios_results(experiment: str, results_list: List):

    admits_df_list = []
    census_df_list = []
    dispositions_df_list = []
    sim_sir_w_date_df_list = []
    vars_df_list = []
    params_dict_list = []
    wide_df_list = []
    long_df_list = []

    for results in results_list:
        scenario = results['scenario']
        mit_date = pd.to_datetime(results['input_params_dict']['mitigation_date'], format="%Y-%m-%d")
        rel_con_rate = results['input_params_dict']['relative_contact_rate']
        inf_days = results['input_params_dict']['infectious_days']

        admits_df = results['admits_df']
        census_df = results['census_df']
        dispositions_df = results['dispositions_df']
        sim_sir_w_date_df = results['sim_sir_w_date_df']
        wide_df = results['adm_cen_wide_df']
        long_df = results['adm_cen_long_df']

        for df, df_list in (
                (sim_sir_w_date_df, sim_sir_w_date_df_list),
                (dispositions_df, dispositions_df_list),
                (admits_df, admits_df_list),
                (census_df, census_df_list),
                (wide_df, wide_df_list),
                (long_df, long_df_list),
        ):
            df['scenario'] = scenario
            df['mit_date'] = mit_date
            df['rel_con_rate'] = rel_con_rate
            df['inf_days'] = inf_days
            df_list.append(df.copy())

        params_dict_list.append(results['input_params_dict'].copy())
        vars_df = pd.DataFrame(results['important_variables_dict'], index=[0])
        vars_df['scenario'] = scenario
        vars_df_list.append(vars_df.copy())

    cons_dfs = {}
    for df_name, df_list in (
            ("sim_sir_w_date_df", sim_sir_w_date_df_list),
            ("dispositions_df", dispositions_df_list),
            ("admits_df", admits_df_list),
            ("census_df", census_df_list),
            ("vars_df", vars_df_list),
            ("adm_cen_wide_df", wide_df_list),
            ("adm_cen_long_df", long_df_list),
    ):
        cons_dfs[df_name] = pd.concat(df_list)

    cons_dfs['vars_df'].reset_index(inplace=True, drop=True)

    return cons_dfs, params_dict_list


def market_share_adjustment(market_share_csv, base_results, mkt_scenario):
    """

    :param market_share_csv:
    :param base_results:
    :param mkt_scenario:
    :return: results dictionary
    """

    # Get the hosp, icu and vent rates from the inputs
    rates = {
        key: d.rate
        for key, d in base_results['input_params_dict']['dispositions'].items()
    }

    days = {
        key: d.days
        for key, d in base_results['input_params_dict']['dispositions'].items()
    }

    # Read market share file
    market_share_df = pd.read_csv(market_share_csv, parse_dates=['date'])

    sim_sir_w_date_df = base_results['sim_sir_w_date_df'].copy()
    all_w_mkt_df = pd.merge(sim_sir_w_date_df, market_share_df, on=['date'], how='left')

    all_w_mkt_df = calculate_dispositions_mkt_adj(all_w_mkt_df, rates)
    all_w_mkt_df = calculate_admits_mkt_adj(all_w_mkt_df, rates)
    all_w_mkt_df = calculate_census_mkt_adj(all_w_mkt_df, days)

    dispositions_mkt_df = pd.DataFrame(data={
        'day': all_w_mkt_df['day'],
        'date': all_w_mkt_df['date'],
        'ever_hospitalized': all_w_mkt_df['ever_hospitalized'],
        'ever_icu': all_w_mkt_df['ever_icu'],
        'ever_ventilated': all_w_mkt_df['ever_ventilated'],
    })
    admits_mkt_df = pd.DataFrame(data={
        'day': all_w_mkt_df['day'],
        'date': all_w_mkt_df['date'],
        'admits_hospitalized': all_w_mkt_df['admits_hospitalized'],
        'admits_icu': all_w_mkt_df['admits_icu'],
        'admits_ventilated': all_w_mkt_df['admits_ventilated'],
    })
    census_mkt_df = pd.DataFrame(data={
        'day': all_w_mkt_df['day'],
        'date': all_w_mkt_df['date'],
        'census_hospitalized': all_w_mkt_df['census_hospitalized'],
        'census_icu': all_w_mkt_df['census_icu'],
        'census_ventilated': all_w_mkt_df['census_ventilated'],
    })

    wide_df, long_df = join_and_melt(admits_mkt_df, census_mkt_df, mkt_scenario)

    base_results['important_variables_dict']['result_type'] = 'postprocessor'

    results_mkt = {
        'result_type': 'postprocessor',
        'scenario': mkt_scenario,
        'input_params_dict': base_results['input_params_dict'],
        'important_variables_dict': base_results['important_variables_dict'],
        'sim_sir_w_date_df': base_results['sim_sir_w_date_df'],
        'dispositions_df': dispositions_mkt_df,
        'admits_df': admits_mkt_df,
        'census_df': census_mkt_df,
        'adm_cen_wide_df': wide_df,
        'adm_cen_long_df': long_df
    }

    return results_mkt


def include_actual(results, actual_csv):
    actual_df = pd.read_csv(actual_csv, parse_dates=['date'])
    wide_df = results['adm_cen_wide_df']
    wide_w_act_df = pd.merge(wide_df, actual_df, left_on=['date'], right_on=['date'], how='left')
    wide_w_act_df.rename({'day_x': 'day'}, axis='columns', inplace=True)
    wide_w_act_df.drop(['day_y'], axis='columns', inplace=True)

    # Now recreate long version. It's assumed that the only columns in actual are
    # day, date, and meltable measures.

    long_df = pd.melt(wide_w_act_df,
                      id_vars=['scenario', 'day', 'date'],
                      var_name='dispo_measure', value_name='cases')

    results['adm_cen_wide_df'] = wide_w_act_df.copy()
    results['adm_cen_long_df'] = long_df

    return results


def calculate_dispositions_mkt_adj(
    mkt_adj_df: pd.DataFrame,
    rates: Dict[str, float],
):
    """Build dispositions dataframe of patients adjusted by rate and market_share."""
    for key, rate in rates.items():
        mkt_adj_df["ever_" + key] = (mkt_adj_df.infected +
                                     mkt_adj_df.recovered) * rate * mkt_adj_df.market_share

    return mkt_adj_df


def calculate_admits_mkt_adj(mkt_adj_df: pd.DataFrame, rates):
    """Build admits dataframe from dispositions."""
    for key in rates.keys():
        # Need to convert Series to ndarray else get weird slicing errors
        # when doing admit[1:] = ever[1:] - ever[:-1]. Strange.
        ever = np.array(mkt_adj_df["ever_" + key])
        admit = np.empty_like(ever)
        admit[0] = np.nan
        admit[1:] = ever[1:] - ever[:-1]
        mkt_adj_df["admits_" + key] = admit

    return mkt_adj_df


def calculate_census_mkt_adj(
    mkt_adj_df: pd.DataFrame,
    lengths_of_stay: Dict[str, int],
):
    """Average Length of Stay for each disposition of COVID-19 case (total guesses)"""
    n_days = mkt_adj_df["day"].shape[0]

    for key, los in lengths_of_stay.items():
        raw = np.array(mkt_adj_df["admits_" + key])
        cumsum = np.empty(n_days + los)
        cumsum[:los+1] = 0.0
        cumsum[los+1:] = raw[1:].cumsum()

        census = cumsum[los:] - cumsum[:-los]
        mkt_adj_df["census_" + key] = census

    return mkt_adj_df


def main():
    my_args = parse_args()
    my_args_dict = vars(my_args)

    scenario = my_args.scenario
    output_path = my_args.output_path

    # Read chime params from configuration file
    p = create_params_from_file(my_args_dict['parameters'])

    input_check = vars(p)

    if my_args.experiment is None:
        # Just running one scenario
        m, results = sim_chime(scenario, p)

        if not my_args.quiet:
            print("Scenario: {}\n".format(results['scenario']))
            print("\nInput parameters")
            print("{}".format(50 * '-'))
            print(json.dumps(results['input_params_dict'], indent=4, sort_keys=False, default=str))

            print("\nIntermediate variables")
            print("{}".format(50 * '-'))
            print(json.dumps(results['important_variables_dict'], indent=4, sort_keys=False))
            print("\n")

        # Check if doing market share adjustments
        if my_args.market_share is not None:

            mkt_scenario = Path(my_args.market_share).stem
            # Make sure mkt_scenario != scenario since will be value in wide and long dfs
            if mkt_scenario == scenario:
                mkt_scenario += '_mkt'
            mkt_share_csv = my_args.market_share
            results_mkt = market_share_adjustment(mkt_share_csv,
                                                  results, mkt_scenario)

            # Stack wide and long results dataframes?
            # This seems ill thought out. Let's not allow it for now.
            if 0:
                for key in ('wide_df', 'long_df'):
                    df = results['adm_cen_' + key]
                    mkt_df = results_mkt['adm_cen_' + key]
                    df = pd.concat([df, mkt_df])
                    results['adm_cen_' + key] = df.copy()

            else:
                write_results(results_mkt, mkt_scenario, output_path)

        # Check if including actual values
        if my_args.actual is not None:
            results = include_actual(results, my_args.actual)

        # Write out results (may or may not be stacked)
        write_results(results, scenario, output_path)

    else:
        # Running a bunch of scenarios using sim_chimes()
        experiment = my_args.experiment
        # Run the scenarios (kludged into sim_chimes for now)
        results_list = sim_chimes(experiment, p)
        # Check if including actual values
        if my_args.actual is not None:
            for results in results_list:
                new_results = include_actual(results, my_args.actual)
                results.update(new_results)

        # Consolidate results over scenarios
        cons_dfs, params_dict_list = consolidate_scenarios_results(experiment, results_list)

        # Write out consolidated csv files
        write_experiment_results(cons_dfs, params_dict_list, experiment, output_path)




if __name__ == "__main__":
    main()


"""
MIT License

Copyright (c) 2020 Mark Isken
Copyright (c) 2020 The Trustees of the University of Pennsylvania

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


