# Copyright 2014 (C) Adam Greig
#
# This file is part of Tawhiri.
#
# Tawhiri is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Tawhiri is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Tawhiri.  If not, see <http://www.gnu.org/licenses/>.

"""
Provide all the balloon models, termination conditions and
functions to combine models and termination conditions.
"""

import calendar
import itertools
import math
import numpy as np
from random import normalvariate, uniform

from . import interpolate
from .dataset import Dataset


_PI_180 = math.pi / 180.0
_180_PI = 180.0 / math.pi

R_air = 287.05  # Gas constant of air [J/kg-K]
R_helium = 2077.1  # Gas constant of helium [J/kg-K]

MB_TO_PA = 100  # millibar to Pascal multiplicative conversion factor

g0 = 9.80665  # Standard acceleration of gravity [m/s^2]


## Up/Down Models #############################################################


def make_constant_ascent(ascent_rate):
    """Return a constant-ascent model at `ascent_rate` (m/s)"""
    def constant_ascent(t, lat, lng, alt):
        return 0.0, 0.0, ascent_rate
    return constant_ascent

def make_bpp_ascent(dataset, warningcounts, helium_mass, system_mass, dataset_errors=None):
    """
        Return a 3-D ascent model that models ascent rate based on dataset temperature,
        pressure, and altitude, and the balloon's altitude-varying radius.
    """
    get_atmospheric_state = interpolate.make_interpolator(dataset,
                                                          warningcounts,
                                                          dataset_errors)
    dataset_epoch = calendar.timegm(dataset.ds_time.timetuple())

    def state_function(t, lat, lng, alt):
        t -= dataset_epoch
        u, v, temperature, pressure = get_atmospheric_state(t / 3600.0, lat, lng, alt)
        pressure = pressure * MB_TO_PA  # convert to consistent units
        rho_air = pressure/R_air/temperature
        rho_helium = pressure/R_helium/temperature  # temp and pressure inside balloon assumed equal to outside
        balloon_radius = math.pow(3.0*helium_mass/(4.0*math.pi*rho_helium), 1.0/3.0)
        numerator = ((4.0/3.0)*math.pi*math.pow(balloon_radius, 3)*(rho_air - rho_helium) - system_mass)*g0
        denominator = 0.5*rho_air*drag_coefficient(rho_air, balloon_radius, temperature, 0)*math.pi*math.pow(balloon_radius, 2)
        w = math.sqrt(numerator/denominator)
        h = 6371009 + alt
        dlat = _180_PI * v / h
        dlng = _180_PI * u / (h * math.cos(lat * _PI_180))
        return dlat, dlng, w
    return state_function


def make_drag_descent(sea_level_descent_rate):
    """Return a descent-under-parachute model with sea level descent
       `sea_level_descent_rate` (m/s). Descent rate at altitude is determined
       using an altitude model courtesy of NASA:
       http://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html

       For a given altitude the air density is computed, a drag coefficient is
       estimated from the sea level descent rate, and the resulting terminal
       velocity is computed by the returned model function.
    """
    def density(alt):
        temp = pressure = 0.0
        if alt > 25000:
            temp = -131.21 + 0.00299 * alt
            pressure = 2.488 * ((temp + 273.1)/(216.6)) ** (-11.388)
        elif 11000 < alt <= 25000:
            temp = -56.46
            pressure = 22.65 * math.exp(1.73 - 0.000157 * alt)
        else:
            temp = 15.04 - 0.00649 * alt
            pressure = 101.29 * ((temp + 273.1)/288.08) ** (5.256)
        return pressure / (0.2869*(temp + 273.1))

    drag_coefficient = sea_level_descent_rate * 1.1045

    def drag_descent(t, lat, lng, alt):
        return 0.0, 0.0, -drag_coefficient/math.sqrt(density(alt))
    return drag_descent


## Sideways Models ############################################################


def make_wind_velocity(dataset, warningcounts, dataset_errors=None):
    """Return a wind-velocity model, which gives lateral movement at
       the wind velocity for the current time, latitude, longitude and
       altitude. The `dataset` argument is the wind dataset in use.
    """
    get_atmospheric_state = interpolate.make_interpolator(dataset,
                                                          warningcounts,
                                                          dataset_errors)
    dataset_epoch = calendar.timegm(dataset.ds_time.timetuple())
    def wind_velocity(t, lat, lng, alt):
        t -= dataset_epoch
        u, v = get_atmospheric_state(t / 3600.0, lat, lng, alt)[0:2]
        R = 6371009 + alt
        dlat = _180_PI * v / R
        dlng = _180_PI * u / (R * math.cos(lat * _PI_180))
        return dlat, dlng, 0.0
    return wind_velocity


## Termination Criteria #######################################################


def make_burst_termination(burst_altitude):
    """Return a burst-termination criteria, which terminates integration
       when the altitude reaches `burst_altitude`.
    """
    def burst_termination(t, lat, lng, alt):
        if alt >= burst_altitude:
            return True
    return burst_termination


def sea_level_termination(t, lat, lng, alt):
    """A termination criteria which terminates integration when
       the altitude is less than (or equal to) zero.

       Note that this is not a model factory.
    """
    if alt <= 0:
        return True

def make_elevation_data_termination(dataset=None):
    """A termination criteria which terminates integration when the
       altitude goes below ground level, using the elevation data
       in `dataset` (which should be a ruaumoko.Dataset).
    """
    def tc(t, lat, lng, alt):
        return dataset.get(lat, lng) > alt
    return tc

def make_time_termination(max_time):
    """A time based termination criteria, which terminates integration when
       the current time is greater than `max_time` (a UNIX timestamp).
    """
    def time_termination(t, lat, lng, alt):
        if t > max_time:
            return True
    return time_termination


## Model Combinations #########################################################


def make_linear_model(models):
    """Return a model that returns the sum of all the models in `models`.
    """
    def linear_model(t, lat, lng, alt):
        dlat, dlng, dalt = 0.0, 0.0, 0.0
        for model in models:
            d = model(t, lat, lng, alt)
            dlat, dlng, dalt = dlat + d[0], dlng + d[1], dalt + d[2]
        return dlat, dlng, dalt
    return linear_model


def make_any_terminator(terminators):
    """Return a terminator that terminates when any of `terminators` would
       terminate.
    """
    def terminator(t, lat, lng, alt):
        return any(term(t, lat, lng, alt) for term in terminators)
    return terminator


## Pre-Defined Profiles #######################################################


def standard_profile_cusf(ascent_rate, burst_altitude, descent_rate,
                          wind_dataset, elevation_dataset, warningcounts,
                          ascent_rate_std_dev=0, burst_altitude_std_dev=0,
                          descent_rate_std_dev=0, wind_std_dev=0):
    """Make a model chain for the standard high altitude balloon situation of
       ascent at a constant rate followed by burst and subsequent descent
       at terminal velocity under parachute with a predetermined sea level
       descent rate.

       Requires the balloon `ascent_rate`, `burst_altitude` and `descent_rate`,
       and additionally requires the dataset to use for wind velocities.

       Returns a tuple of (model, terminator) pairs.
    """

    ascent_rate = normalvariate(ascent_rate, ascent_rate_std_dev)
    burst_altitude = normalvariate(burst_altitude, burst_altitude_std_dev)
    descent_rate = normalvariate(descent_rate, descent_rate_std_dev)

    dataset_error = generate_dataset_error(wind_std_dev)

    model_up = make_linear_model([make_constant_ascent(ascent_rate),
                                  make_wind_velocity(wind_dataset,
                                                     warningcounts,
                                                     dataset_error)])
    term_up = make_burst_termination(burst_altitude)

    model_down = make_linear_model([make_drag_descent(descent_rate),
                                    make_wind_velocity(wind_dataset,
                                                       warningcounts,
                                                       dataset_error)])
    term_down = make_elevation_data_termination(elevation_dataset)

    return ((model_up, term_up), (model_down, term_down))


def standard_profile_bpp(helium_mass, dry_mass, burst_altitude,
                         sea_level_descent_rate, wind_dataset, elevation_dataset,
                         warningcounts, burst_altitude_std_dev=0,
                         descent_rate_std_dev=0, wind_std_dev=0,
                         helium_mass_std_dev = 0):
    """
    Make a model chain for the standard high altitude balloon situation of ascent until burst and
    descent under parachute. Ascent rate is calculated using the BPP physics model, which calculates
    the ascent velocity using the balloon's time-varying radius and the density computed from the
    weather model data.
    :param helium_mass: The mass of helium in the balloon, in kg
    :param dry_mass: The mass of payload and balloon, in kg
    :param burst_altitude: The burst altitude, in m
    :param sea_level_descent_rate: The descent rate of the system at sea level, in m/s
    :param wind_dataset: The wind dataset to use
    :param elevation_dataset: The ruaumoko elevation dataset to use
    :param warningcounts: The warningcounts object to use
    :param burst_altitude_std_dev: The standard deviation in burst altitude to use for Monte Carlo runs, in m
    :param descent_rate_std_dev: The standard deviation in sea level descent rate to use for Monte Carlo runs, in m/s
    :param wind_std_dev: The standard deviation in wind magnitudes to use, as a fraction
    :param helium_mass_std_dev The standard deviation for helium mass to use in Monte Carlo runs, in kg
    :return: A tuple of (model, terminator) pairs representing the stages of the flight
    """

    burst_altitude = normalvariate(burst_altitude, burst_altitude_std_dev)
    sea_level_descent_rate = normalvariate(sea_level_descent_rate, descent_rate_std_dev)
    helium_mass = normalvariate(helium_mass, helium_mass_std_dev)

    dataset_error = generate_dataset_error(wind_std_dev)

    model_up = make_bpp_ascent(wind_dataset, warningcounts,
                               helium_mass, dry_mass,
                               dataset_error)

    term_up = make_burst_termination(burst_altitude)

    model_down = make_linear_model([make_drag_descent(sea_level_descent_rate),
                                    make_wind_velocity(wind_dataset,
                                                       warningcounts,
                                                       dataset_error)])
    term_down = make_elevation_data_termination(elevation_dataset)

    return ((model_up, term_up), (model_down, term_down))


def float_profile(ascent_rate, float_altitude, stop_time, dataset, warningcounts):
    """Make a model chain for the typical floating balloon situation of ascent
       at constant altitude to a float altitude which persists for some
       amount of time before stopping. Descent is in general not modelled.
    """

    model_up = make_linear_model([make_constant_ascent(ascent_rate),
                                  make_wind_velocity(dataset, warningcounts)])
    term_up = make_burst_termination(float_altitude)
    model_float = make_wind_velocity(dataset, warningcounts)
    term_float = make_time_termination(stop_time)

    return ((model_up, term_up), (model_float, term_float))

## Support Functions ##########################################################

def generate_dataset_error(max_wind_deviation):
    dataset_error = np.zeros((Dataset.NUM_GFS_VARIABLES,
                            Dataset.NUM_GFS_LAT_STEPS,
                            Dataset.NUM_GFS_LNG_STEPS))

    for var_index, lat_index, lng_index in itertools.product(
            range(1, Dataset.NUM_GFS_VARIABLES),  # TODO magic number
            range(Dataset.NUM_GFS_LAT_STEPS),
            range(Dataset.NUM_GFS_LNG_STEPS)):
        dataset_error[var_index, lat_index, lng_index] =\
            uniform(-max_wind_deviation, max_wind_deviation)

    return dataset_error

def drag_coefficient(rho_air, balloon_radius, temperature, velocity):
    return 0.5  # TODO should perform computation based on Reynolds #
