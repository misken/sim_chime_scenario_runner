"""
Extending base sir to accept dynamic relative contact rates.

This becomes important in later stages of epidemic as we move beyond first peak and
regions start to relax social distancing policies.

For this first attempt at allowing multiple mitigation dates:
- date-first-hospitalized is required as is an initial doubling time. These can be estimated from actual data and
CHIME runs during early phase of epidemic.
- the policy variable should contain (beta, n_days) pairs for all days from date-first-hospitalized on
- final (beta, n_days) pair is assumed to go on forever
"""

from datetime import datetime, timedelta
from logging import INFO, basicConfig, getLogger
from sys import stdout
from typing import Dict, Tuple, Sequence, Optional

import numpy as np
import pandas as pd

from penn_chime.model.parameters import Parameters
from penn_chime.model.sir import Sir
from penn_chime.model.sir import sim_sir
from penn_chime.model.sir import calculate_admits, calculate_census, calculate_dispositions
from penn_chime.model.sir import build_floor_df, build_sim_sir_w_date_df
from penn_chime.model.sir import get_growth_rate


basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=stdout,
)
logger = getLogger(__name__)


class SirPlus(Sir):

    def __init__(self, p: Parameters,
                 intrinsic_growth_rate: float = None,
                 initial_doubling_time: float = None,
                 admits_df: Optional = None,
                 rcr_policies_df: Optional = None):
        """
        Main model class for using dynamic relative contact rates (rcr).

        :param p:
        :param intrinsic_growth_rate:
        :param initial_doubling_time:
        :param admits_df:
        :param rcr_policies_df:
        """

        self.rates = {
            key: d.rate
            for key, d in p.dispositions.items()
        }

        self.days = {
            key: d.days
            for key, d in p.dispositions.items()
        }

        self.keys = ("susceptible", "infected", "recovered")

        # An estimate of the number of infected people on the day that
        # the first hospitalized case is seen
        #
        # Note: this should not be an integer.
        # Use default hospitalized.rate from input file. Will allos
        # updated estimate based on admits later for projecting
        # admissions from sir output.
        infected = (
            1.0 / p.market_share / p.hospitalized.rate
        )

        susceptible = p.population - infected

        gamma = 1.0 / p.infectious_days
        self.gamma = gamma

        self.susceptible = susceptible
        self.infected = infected
        self.recovered = p.recovered

        # Get first hospitalized date either from admits data for parameters
        if admits_df is not None:
            date_first_hospitalized_ts = admits_df[admits_df.iloc[:, 1] > 0].iloc[0, 0]
            date_first_hospitalized = date_first_hospitalized_ts.date()
        else:
            date_first_hospitalized = p.date_first_hospitalized

        self.i_day = (p.current_date - date_first_hospitalized).days
        self.current_hospitalized = p.current_hospitalized
        logger.info(
            'Using date_first_hospitalized: %s; current_date: %s; i_day: %s, current_hospitalized: %s',
            date_first_hospitalized,
            p.current_date,
            self.i_day,
            p.current_hospitalized,
        )

        if intrinsic_growth_rate is None and p.doubling_time is None:
            logger.info(
                'doubling_time: %s; date_first_hospitalized: %s',
                p.doubling_time,
                p.date_first_hospitalized,
            )
            raise AssertionError('doubling_time or admit data file must be provided if using dynamic relative contact '
                                 'rates.')

        if intrinsic_growth_rate is None:
            # intrinsic growth rate based on first user input of initial doubling_time
            self.intrinsic_growth_rate = get_growth_rate(p.doubling_time)
            self.initial_doubling_time = get_doubling_time(self.intrinsic_growth_rate)
        else:
            self.intrinsic_growth_rate = intrinsic_growth_rate
            self.initial_doubling_time = initial_doubling_time

        # Generate beta policies. Will use p.mitigation_date if rcr_policies is None
        self.beta = get_beta(self.intrinsic_growth_rate, self.gamma, self.susceptible, 0.0)
        self.r_naught = self.beta / gamma * susceptible
        if rcr_policies_df is None:
            self.beta_t = get_beta(intrinsic_growth_rate, self.gamma, self.susceptible, p.relative_contact_rate)
            self.beta_policies = self.get_betas(p)
        else:
            self.beta_policies = self.get_betas(p, rcr_policies_df)

        # Run the projections now that we have beta policies
        self.raw = self.run_projection(p, self.beta_policies)
        self.population = p.population

        self.raw["date"] = self.raw["day"].astype("timedelta64[D]") + np.datetime64(p.current_date)

        self.raw_df = pd.DataFrame(data=self.raw)
        self.dispositions_df = pd.DataFrame(data={
            'day': self.raw['day'],
            'date': self.raw['date'],
            'ever_hospitalized': self.raw['ever_hospitalized'],
            'ever_icu': self.raw['ever_icu'],
            'ever_ventilated': self.raw['ever_ventilated'],
        })
        self.admits_df = pd.DataFrame(data={
            'day': self.raw['day'],
            'date': self.raw['date'],
            'admits_hospitalized': self.raw['admits_hospitalized'],
            'admits_icu': self.raw['admits_icu'],
            'admits_ventilated': self.raw['admits_ventilated'],
        })
        self.census_df = pd.DataFrame(data={
            'day': self.raw['day'],
            'date': self.raw['date'],
            'census_hospitalized': self.raw['census_hospitalized'],
            'census_icu': self.raw['census_icu'],
            'census_ventilated': self.raw['census_ventilated'],
        })

        logger.info('len(np.arange(-i_day, n_days+1)): %s', len(np.arange(-self.i_day, p.n_days+1)))
        logger.info('len(raw_df): %s', len(self.raw_df))

        self.infected = self.raw_df['infected'].values[self.i_day]
        self.susceptible = self.raw_df['susceptible'].values[self.i_day]
        self.recovered = self.raw_df['recovered'].values[self.i_day]

        # We set this at top since we needed it to compute betas
        #self.intrinsic_growth_rate = intrinsic_growth_rate

        # r, betas and doubling times are all dynamic in that not just one mitigation date
        # r_t is r_0 after distancing
        if rcr_policies_df is None:
            self.r_t = self.beta_t / gamma * susceptible

            doubling_time_t = 1.0 / np.log2(
                self.beta_t * susceptible - gamma + 1)

            self.doubling_time_t = doubling_time_t

        self.sim_sir_w_date_df = build_sim_sir_w_date_df(self.raw_df, p.current_date, self.keys)

        self.sim_sir_w_date_floor_df = build_floor_df(self.sim_sir_w_date_df, self.keys, "")
        self.admits_floor_df = build_floor_df(self.admits_df, p.dispositions.keys(), "admits_")
        self.census_floor_df = build_floor_df(self.census_df, p.dispositions.keys(), "census_")

        # self.daily_growth_rate = get_growth_rate(p.doubling_time)
        # self.daily_growth_rate_t = get_growth_rate(self.doubling_time_t)

    def get_betas(self, p, rcr_policies=None):
        if rcr_policies is None:
            if p.mitigation_date is not None:
                mitigation_day = -(p.current_date - p.mitigation_date).days
            else:
                mitigation_day = 0

            total_days = self.i_day + p.n_days

            if mitigation_day < -self.i_day:
                mitigation_day = -self.i_day

            pre_mitigation_days = self.i_day + mitigation_day
            post_mitigation_days = total_days - pre_mitigation_days

            return [
                (self.beta, pre_mitigation_days),
                (self.beta_t, post_mitigation_days),
            ]
        else:
            beta_policies = []
            total_days = self.i_day + p.n_days
            last_date = p.current_date + pd.Timedelta(days=p.n_days - 1)

            # Get date column name and reset index to avoid pandas hell of boolean indexing with date columns
            colname = rcr_policies.columns[0]
            rcr_policies = rcr_policies.set_index(colname)
            rcr_policies = rcr_policies.loc[pd.Timestamp(p.date_first_hospitalized):pd.Timestamp(last_date)]
            rcr_policies.reset_index(inplace=True, drop=False)
            # Compute number of days rcr policy in effect; last value will be nan
            rcr_policies['n_days'] = rcr_policies.iloc[:, 0].diff(-1).apply(lambda x: -x.days)
            # Fill in the last day's nan value with the proper number of days
            tot_policy_days = rcr_policies['n_days'].sum()
            rcr_policies['n_days'].fillna(total_days - tot_policy_days, inplace=True)
            rcr_policies['n_days'] = rcr_policies['n_days'].astype('int64')

            # Convert rcr and n_days values into beta and n_days values to use with current
            # CHIME approach to modeling a single mitigation date.
            for (date, rcr, n_days) in rcr_policies.itertuples(index=False):
                new_beta = get_beta(self.intrinsic_growth_rate, self.gamma, self.susceptible, rcr)
                beta_policies.append((new_beta, n_days))

            return beta_policies

    def run_projection(self, p: Parameters, policy: Sequence[Tuple[float, int]]):
        raw = sim_sir(
            self.susceptible,
            self.infected,
            p.recovered,
            self.gamma,
            -self.i_day,
            policy
        )

        calculate_dispositions(raw, self.rates, p.market_share)
        calculate_admits(raw, self.rates)
        calculate_census(raw, self.days)

        return raw


def get_doubling_time(igr):
    return 1.0 / np.log2(1.0 + igr)


def get_beta(intrinsic_growth_rate: float,
             gamma: float,
             susceptible: float,
             relative_contact_rate: float
             ) -> float:
    return (
            (intrinsic_growth_rate + gamma)
            / susceptible
            * (1.0 - relative_contact_rate)
    )
