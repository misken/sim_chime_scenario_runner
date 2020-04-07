import pytest

from datetime import datetime
from penn_chime.constants import CHANGE_DATE
from penn_chime.parameters import Parameters, Disposition
from penn_chime.models import SimSirModel as Model
import penn_chime.models as models
import penn_chime.cli as cli

from pandas import DataFrame
import numpy as np
import pandas as pd

import sys
import json

import sim_chime_scenario_runner as runner

@pytest.fixture
def admits_dt361_df():
    return pd.read_csv('tests/output/2020-03-27_dt361_projected_admits.csv', parse_dates=['date'])

@pytest.fixture
def census_dt361_df():
    return pd.read_csv('tests/output/2020-03-27_dt361_projected_census.csv', parse_dates=['date'])

@pytest.fixture
def admits_fd20200220_df():
    return pd.read_csv('tests/output/2020-03-27_fd0220_projected_admits.csv', parse_dates=['date'])

@pytest.fixture
def census_fd20200220_df():
    return pd.read_csv('tests/output/2020-03-27_fd0220_projected_census.csv', parse_dates=['date'])


def test_import(admits_dt361_df, census_dt361_df,
                admits_fd20200220_df, census_fd20200220_df):


    p_dt361 = runner.create_params_from_file('tests/dt361.cfg')
    p_fd20200220 = runner.create_params_from_file('tests/fd0220.cfg')

    scenario = 'runner_import_dt361'
    model, results_dt361 = runner.sim_chime(scenario, p_dt361)
    
    # Admits - dt361
    pd.testing.assert_series_equal(results_dt361['admits_df']['hospitalized'],
                                         admits_dt361_df['hospitalized'],
                                         check_less_precise=True)

    pd.testing.assert_series_equal(results_dt361['admits_df']['icu'],
                                         admits_dt361_df['icu'],
                                         check_less_precise=True)

    pd.testing.assert_series_equal(results_dt361['admits_df']['ventilated'],
                                         admits_dt361_df['ventilated'],
                                         check_less_precise=True)

    pd.testing.assert_series_equal(results_dt361['admits_df']['day'],
                                   admits_dt361_df['day'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_dt361['admits_df']['date'],
                                   admits_dt361_df['date'],
                                   check_less_precise=True)

    # Census - dt361
    pd.testing.assert_series_equal(results_dt361['census_df']['hospitalized'],
                                         census_dt361_df['hospitalized'],
                                         check_less_precise=True)

    pd.testing.assert_series_equal(results_dt361['census_df']['icu'],
                                         census_dt361_df['icu'],
                                         check_less_precise=True)

    pd.testing.assert_series_equal(results_dt361['census_df']['ventilated'],
                                         census_dt361_df['ventilated'],
                                         check_less_precise=True)

    pd.testing.assert_series_equal(results_dt361['census_df']['day'],
                                   census_dt361_df['day'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_dt361['census_df']['date'],
                                   census_dt361_df['date'],
                                   check_less_precise=True)

    scenario = 'runner_import_fd20200220'
    model, results_fd20200220 = runner.sim_chime(scenario, p_fd20200220)

    # Admits - fd20200220
    pd.testing.assert_series_equal(results_fd20200220['admits_df']['hospitalized'],
                                   admits_fd20200220_df['hospitalized'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_fd20200220['admits_df']['icu'],
                                   admits_fd20200220_df['icu'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_fd20200220['admits_df']['ventilated'],
                                   admits_fd20200220_df['ventilated'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_fd20200220['admits_df']['day'],
                                   admits_fd20200220_df['day'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_fd20200220['admits_df']['date'],
                                   admits_fd20200220_df['date'],
                                   check_less_precise=True)

    # Census - fd20200220
    pd.testing.assert_series_equal(results_fd20200220['census_df']['hospitalized'],
                                   census_fd20200220_df['hospitalized'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_fd20200220['census_df']['icu'],
                                   census_fd20200220_df['icu'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_fd20200220['census_df']['ventilated'],
                                   census_fd20200220_df['ventilated'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_fd20200220['census_df']['day'],
                                   census_fd20200220_df['day'],
                                   check_less_precise=True)

    pd.testing.assert_series_equal(results_fd20200220['census_df']['date'],
                                   census_fd20200220_df['date'],
                                   check_less_precise=True)

    print(results_dt361['scenario'])
    print(results_dt361['input_params_dict'])
    print(results_dt361['intermediate_variables_dict'])

    print(results_fd20200220['scenario'])
    print(results_fd20200220['input_params_dict'])
    print(results_fd20200220['intermediate_variables_dict'])