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

DEFAULT_VELOCITY = 6  # typical ascent velocity to use as basis for iteration
VELOCITY_TOLERANCE = 0.05  # Tolerance for Reynolds number calculations

DATASET_ERROR_VARIABLES = 2  # Wind magnitude and velocity

g0 = 9.80665  # Standard acceleration of gravity [m/s^2]

VAR_WIND_MAG_ERROR = 0
VAR_WIND_ANGLE_ERROR = 1


## Up/Down Models #############################################################


def make_constant_ascent(ascent_rate):
    """Return a constant-ascent model at `ascent_rate` (m/s)"""
    def constant_ascent(t, lat, lng, alt, t_film, t_gas):
        return 0.0, 0.0, ascent_rate, 0.0, 0.0
    return constant_ascent

def make_bpp_ascent(dataset, warningcounts, helium_mass, system_mass, dataset_errors=None):
    """
        Return a 3-D ascent model that models ascent rate based on dataset temperature,
        pressure, and altitude, and the balloon's altitude-varying radius.
    """
    get_atmospheric_state = interpolate.make_interpolator(dataset,
                                                          warningcounts)
    dataset_epoch = calendar.timegm(dataset.ds_time.timetuple())

    def state_function(t, lat, lng, alt, t_film, t_gas):
        t -= dataset_epoch
        u, v, temperature, pressure = get_atmospheric_state(t / 3600.0, lat, lng, alt)
        if dataset_errors is not None:
            u, v = vary_wind_velocity(u, v, lat, lng, dataset_errors)
        pressure = pressure * MB_TO_PA  # convert to consistent units
        rho_air = pressure/R_air/temperature
        rho_helium = pressure/R_helium/temperature  # TODO change to t_gas
        balloon_radius = math.pow(3.0*helium_mass/(4.0*math.pi*rho_helium), 1.0/3.0)
        w = DEFAULT_VELOCITY
        while True:
            w_old = w
            numerator = ((4.0 / 3.0) * math.pi * math.pow(balloon_radius, 3) * (
                        rho_air - rho_helium) - system_mass) * g0
            denominator = 0.5 * rho_air * drag_coefficient(rho_air, balloon_radius, temperature,
                                                           DEFAULT_VELOCITY) * math.pi * math.pow(balloon_radius, 2)
            w = math.sqrt(numerator / denominator)
            if abs(w - w_old) < VELOCITY_TOLERANCE:
                break

        h = 6371009 + alt
        dlat = _180_PI * v / h
        dlng = _180_PI * u / (h * math.cos(lat * _PI_180))
        return dlat, dlng, w, 0.0, 0.0  # TODO add temperature math
    return state_function


def make_drag_descent(dataset, warningcounts, sea_level_descent_rate, dataset_errors=None):
    """
    Returns a descent-under-parachute model assuming terminal velocity at local density.

    The sea-level descent rate is used to compute the ballistic coefficient, assuming a standard
    density of 1.225 kg/m^3. Local density is computed from the atmospheric dataset using the ideal
    gas law, which is used to compute the downward velocity of the balloon.

    This model affects the z component only.

    :param dataset: The atmospheric dataset to use
    :param warningcounts: The warningcounts object to use
    :param sea_level_descent_rate: The balloon's descent rate at sea level in a standard atmosphere
    :return: The parachute descent model, suitable for use in the solver
    """

    get_atmospheric_state = interpolate.make_interpolator(dataset,
                                                          warningcounts)
    dataset_epoch = calendar.timegm(dataset.ds_time.timetuple())

    def density(t, lat, lng, alt):
        t -= dataset_epoch
        temperature, pressure = get_atmospheric_state(t / 3600.0, lat, lng, alt)[2:4]
        pressure = pressure * MB_TO_PA  # convert to consistent units
        return pressure/R_air/temperature

    # This is actually sqrt(2*g0*BC), where BC is the canonical ballistic coefficient of M/C_d/A
    ballistic_coefficient = sea_level_descent_rate * 1.1068  # Sea-level density of 1.225 kg/m^3

    def drag_descent(t, lat, lng, alt, t_film, t_gas):
        return 0.0, 0.0, -ballistic_coefficient/math.sqrt(density(t, lat, lng, alt)), 0.0, 0.0
    return drag_descent


## Sideways Models ############################################################


def make_wind_velocity(dataset, warningcounts, dataset_errors=None):
    """Return a wind-velocity model, which gives lateral movement at
       the wind velocity for the current time, latitude, longitude and
       altitude. The `dataset` argument is the wind dataset in use.
    """
    get_atmospheric_state = interpolate.make_interpolator(dataset,
                                                          warningcounts)  # TODO dataset errors
    dataset_epoch = calendar.timegm(dataset.ds_time.timetuple())
    def wind_velocity(t, lat, lng, alt, t_film, t_gas):
        t -= dataset_epoch
        u, v = get_atmospheric_state(t / 3600.0, lat, lng, alt)[0:2]
        if dataset_errors is not None:
            u, v = vary_wind_velocity(u, v, lat, lng, dataset_errors)
        R = 6371009 + alt
        dlat = _180_PI * v / R
        dlng = _180_PI * u / (R * math.cos(lat * _PI_180))
        return dlat, dlng, 0.0, 0.0, 0.0
    return wind_velocity


## Termination Criteria #######################################################


def make_burst_termination(burst_altitude):
    """Return a burst-termination criteria, which terminates integration
       when the altitude reaches `burst_altitude`.
    """
    def burst_termination(t, lat, lng, alt, t_film, t_gas):
        if alt >= burst_altitude:  # TODO return False otherwise?
            return True
    return burst_termination


def sea_level_termination(t, lat, lng, alt, t_film, t_gas):
    """A termination criteria which terminates integration when
       the altitude is less than (or equal to) zero.

       Note that this is not a model factory.
    """
    if alt <= 0:  # TODO return False otherwise?
        return True

def make_elevation_data_termination(dataset=None):
    """A termination criteria which terminates integration when the
       altitude goes below ground level, using the elevation data
       in `dataset` (which should be a ruaumoko.Dataset).
    """
    def tc(t, lat, lng, alt, t_film, t_gas):
        return dataset.get(lat, lng) > alt
    return tc

def make_time_termination(max_time):
    """A time based termination criteria, which terminates integration when
       the current time is greater than `max_time` (a UNIX timestamp).
    """
    def time_termination(t, lat, lng, alt, t_film, t_gas):
        if t > max_time:
            return True
    return time_termination


def make_diameter_termination(dataset, warningcounts, helium_mass, burst_diameter, dataset_errors=None):

    get_atmospheric_state = interpolate.make_interpolator(dataset,
                                                          warningcounts)
    dataset_epoch = calendar.timegm(dataset.ds_time.timetuple())

    def terminator_function(t, lat, lng, alt, t_film, t_gas):
        t -= dataset_epoch
        temperature, pressure = get_atmospheric_state(t / 3600.0, lat, lng, alt)[2:4]
        pressure = pressure * MB_TO_PA  # convert to consistent units
        rho_helium = pressure/R_helium/temperature  # temp and pressure inside balloon assumed equal to outside
        balloon_radius = math.pow(3.0*helium_mass/(4.0*math.pi*rho_helium), 1.0/3.0)
        return 2.0*balloon_radius > burst_diameter
    return terminator_function


## Model Combinations #########################################################


def make_linear_model(models):
    """Return a model that returns the sum of all the models in `models`.
    """
    def linear_model(t, lat, lng, alt, t_film, t_gas):
        dlat, dlng, dalt, dt_film, dt_gas = 0.0, 0.0, 0.0, 0.0, 0.0
        for model in models:
            d = model(t, lat, lng, alt, t_film, t_gas)
            dlat, dlng, dalt, dt_film, dt_gas = dlat + d[0], dlng + d[1], dalt + d[2], dt_film + d[3], dt_gas + d[4]
        return dlat, dlng, dalt, dt_film, dt_gas
    return linear_model


def make_any_terminator(terminators):
    """Return a terminator that terminates when any of `terminators` would
       terminate.
    """
    def terminator(t, lat, lng, alt, t_film, t_gas):
        return any(term(t, lat, lng, alt, t_film, t_gas) for term in terminators)
    return terminator


## Pre-Defined Profiles #######################################################


def standard_profile_cusf(ascent_rate, burst_altitude, descent_rate,
                          wind_dataset, elevation_dataset, warningcounts,
                          ascent_rate_std_dev=0, burst_altitude_std_dev=0,
                          descent_rate_std_dev=0, wind_mag_std_dev=0,
                          wind_azimuth_std_dev=0):
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

    if wind_mag_std_dev != 0 or wind_azimuth_std_dev != 0:
        dataset_error = generate_dataset_error(wind_mag_std_dev, wind_azimuth_std_dev)
    else:
        dataset_error = None

    model_up = make_linear_model([make_constant_ascent(ascent_rate),
                                  make_wind_velocity(wind_dataset,
                                                     warningcounts,
                                                     dataset_error)])
    term_up = make_burst_termination(burst_altitude)

    model_down = make_linear_model([make_drag_descent(wind_dataset,
                                                      warningcounts,
                                                      descent_rate,
                                                      dataset_error),
                                    make_wind_velocity(wind_dataset,
                                                       warningcounts,
                                                       dataset_error)])
    term_down = make_elevation_data_termination(elevation_dataset)

    return ((model_up, term_up), (model_down, term_down))


def standard_profile_bpp(helium_mass, dry_mass, burst_altitude,
                         sea_level_descent_rate, wind_dataset, elevation_dataset,
                         warningcounts, burst_altitude_std_dev=0,
                         descent_rate_std_dev=0, wind_mag_std_dev=0,
                         wind_azimuth_std_dev=0, helium_mass_std_dev = 0):
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
    :param elevation_dataset: The ruaumoko elevation dataset to use.
    :param warningcounts: The warningcounts object to use
    :param burst_altitude_std_dev: The standard deviation in burst altitude to use for Monte Carlo runs, in m
    :param descent_rate_std_dev: The standard deviation in sea level descent rate to use for Monte Carlo runs, in m/s
    :param wind_mag_std_dev: The standard deviation in wind magnitudes to use, as a fraction
    :param wind_azimuth_std_dev: The standard deviation in wind azimuth to use, in radians
    :param helium_mass_std_dev The standard deviation for helium mass to use in Monte Carlo runs, in kg
    :return: A tuple of (model, terminator) pairs representing the stages of the flight
    """

    burst_altitude = normalvariate(burst_altitude, burst_altitude_std_dev)
    sea_level_descent_rate = normalvariate(sea_level_descent_rate, descent_rate_std_dev)
    helium_mass = normalvariate(helium_mass, helium_mass_std_dev)

    if wind_mag_std_dev != 0 or wind_azimuth_std_dev != 0:
        dataset_error = generate_dataset_error(wind_mag_std_dev, wind_azimuth_std_dev)
    else:
        dataset_error = None

    model_up = make_bpp_ascent(wind_dataset, warningcounts,
                               helium_mass, dry_mass,
                               dataset_error)

    term_up = make_burst_termination(burst_altitude)

    model_down = make_linear_model([make_drag_descent(wind_dataset,
                                                      warningcounts,
                                                      sea_level_descent_rate,
                                                      dataset_error),
                                    make_wind_velocity(wind_dataset,
                                                       warningcounts,
                                                       dataset_error)])
    term_down = make_elevation_data_termination(elevation_dataset)

    return ((model_up, term_up), (model_down, term_down))


def standard_profile_bpp_diameter_term(helium_mass, dry_mass, burst_diameter,
                                       sea_level_descent_rate, wind_dataset, elevation_dataset,
                                       warningcounts, burst_diameter_std_dev=0,
                                       descent_rate_std_dev=0,  wind_mag_std_dev=0,
                                       wind_azimuth_std_dev=0, helium_mass_std_dev=0):
    """
    Make a model chain for the standard high altitude balloon situation of ascent until burst and
    descent under parachute. Ascent rate is calculated using the BPP physics model, which calculates
    the ascent velocity using the balloon's time-varying radius and the density computed from the
    weather model data. Burst is based on the balloon's diameter
    :param helium_mass: The mass of helium in the balloon, in kg
    :param dry_mass: The mass of payload and balloon, in kg
    :param burst_diameter: The burst diameter of the balloon, in m
    :param sea_level_descent_rate: The descent rate of the system at sea level, in m/s
    :param wind_dataset: The wind dataset to use
    :param elevation_dataset: The ruaumoko elevation dataset to use
    :param warningcounts: The warningcounts object to use
    :param burst_diameter_std_dev: The standard deviation in balloon burst diameter to use for Monte Carlo runs, in m
    :param descent_rate_std_dev: The standard deviation in sea level descent rate to use for Monte Carlo runs, in m/s
    :param wind_mag_std_dev: The standard deviation in wind magnitudes to use, as a fraction
    :param wind_azimuth_std_dev: The standard deviation in wind azimuth to use, in radians
    :param helium_mass_std_dev The standard deviation for helium mass to use in Monte Carlo runs, in kg
    :return: A tuple of (model, terminator) pairs representing the stages of the flight
    """

    burst_diameter = normalvariate(burst_diameter, burst_diameter_std_dev)
    sea_level_descent_rate = normalvariate(sea_level_descent_rate, descent_rate_std_dev)
    helium_mass = normalvariate(helium_mass, helium_mass_std_dev)

    dataset_error = generate_dataset_error(wind_mag_std_dev, wind_azimuth_std_dev)

    model_up = make_bpp_ascent(wind_dataset, warningcounts,
                               helium_mass, dry_mass,
                               dataset_error)

    term_up = make_diameter_termination(wind_dataset, warningcounts, helium_mass, burst_diameter, dataset_error)

    model_down = make_linear_model([make_drag_descent(wind_dataset,
                                                      warningcounts,
                                                      sea_level_descent_rate,
                                                      dataset_error),
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

def generate_dataset_error(wind_mag_deviation, wind_angle_deviation):
    dataset_error = np.zeros((DATASET_ERROR_VARIABLES,
                            Dataset.NUM_GFS_LAT_STEPS,
                            Dataset.NUM_GFS_LNG_STEPS))

    for lat_index, lng_index in itertools.product(
            range(Dataset.NUM_GFS_LAT_STEPS),
            range(Dataset.NUM_GFS_LNG_STEPS)):
        dataset_error[VAR_WIND_MAG_ERROR, lat_index, lng_index] =\
            normalvariate(1, wind_mag_deviation)

        dataset_error[VAR_WIND_ANGLE_ERROR, lat_index, lng_index] =\
            normalvariate(0, wind_angle_deviation)

    return dataset_error

def drag_coefficient(rho_air, balloon_radius, temperature, ascent_velocity):
    """
    Returns the coefficient of drag of the balloon, calculated from Conner's drag model
    for Reynolds numbers between 100e3 and 1.2e6. Constant drag is assumed outside of that
    region.
    Ref: J. P. Conner and A. S. Arena, "Near Space Balloon Performance Predictions",
        48th AIAA Aerospace Sciences Meeting (Jan. 2010). doi: 10.2514/6.2010-37
    :param rho_air: air density, in kg/m^3
    :param balloon_radius: radius of the balloon, in m
    :param temperature:
    :param ascent_velocity: ascent velocity, in m/s
    :return: the drag coefficient of the balloon
    """
    reynolds_number = 2*rho_air*balloon_radius*ascent_velocity/air_viscosity(temperature)
    if reynolds_number < 100e3:
        cd = 0.4982609  # to ensure continuity at Re = 100e3
    elif reynolds_number < 1.2e6:
        cd = 7.119e-1 - 2.568e-6*reynolds_number + 4.707e-12*math.pow(reynolds_number, 2) \
            - 4.04e-18*math.pow(reynolds_number, 3) + 1.309e-24*math.pow(reynolds_number, 4)
    else:
        cd = 0.1416024  # to ensure continuity at Re = 1.2e6
    return cd


def air_viscosity(temperature):
    """
    Calculate the dynamic viscosity of air using Sutherland's Formula.
    :param temperature: the air temperature, in K
    :return: the dynamic viscosity, in kg/m-s
    """
    return 1.458e-6*math.pow(temperature, 1.5)/(temperature + 110.4)

def vary_wind_velocity(u_nominal, v_nominal, lat, lng, dataset_errors):
    """
    TODO
    :param u_nominal:
    :param v_nominal:
    :param lat:
    :param lng:
    :param dataset_errors:
    :return:
    """
    lat_index, lat_interpolation_weight = get_lat_lng_index(-90.0, 0.5, lat)
    lng_index, lng_interpolation_weight = get_lat_lng_index(0, 0.5, lng)

    # multiplicative error
    mag_error = dataset_errors[VAR_WIND_MAG_ERROR, lat_index, lng_index] * ( 1 - lat_interpolation_weight) * (1 - lng_interpolation_weight) + \
        dataset_errors[VAR_WIND_MAG_ERROR, lat_index + 1, lng_index] * lat_interpolation_weight * (1 - lng_interpolation_weight) + \
        dataset_errors[VAR_WIND_MAG_ERROR, lat_index, lng_index + 1] * (1 - lat_interpolation_weight) * lng_interpolation_weight + \
        dataset_errors[VAR_WIND_MAG_ERROR, lat_index + 1, lng_index + 1] * lat_interpolation_weight * lng_interpolation_weight

    # in radians
    angle_error = dataset_errors[VAR_WIND_ANGLE_ERROR, lat_index, lng_index] * ( 1 - lat_interpolation_weight) * (1 - lng_interpolation_weight) + \
        dataset_errors[VAR_WIND_ANGLE_ERROR, lat_index + 1, lng_index] * lat_interpolation_weight * (1 - lng_interpolation_weight) + \
        dataset_errors[VAR_WIND_ANGLE_ERROR, lat_index, lng_index + 1] * (1 - lat_interpolation_weight) * lng_interpolation_weight + \
        dataset_errors[VAR_WIND_ANGLE_ERROR, lat_index + 1, lng_index + 1] * lat_interpolation_weight * lng_interpolation_weight

    u = mag_error*(u_nominal*math.cos(angle_error) - v_nominal*math.sin(angle_error))
    v = mag_error*(u_nominal*math.sin(angle_error) + v_nominal*math.cos(angle_error))

    return u, v

def get_lat_lng_index(start, step_size, value):
    """
    TODO
    :param start:
    :param step_size:
    :param value:
    :return:
    """
    a = (value - start) / step_size
    index = math.floor(a)
    interpolation_weight = a - index  # discard integer part (characteristic)
    return index, interpolation_weight


