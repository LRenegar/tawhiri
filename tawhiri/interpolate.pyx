# Copyright 2014 (C) Adam Greig, Daniel Richman
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

# Cython compiler directives:
#
# cython: language_level=3
#
# pick(...) is careful in what it returns:
# cython: boundscheck=False
# cython: wraparound=False
#
# We check for division by zero, and don't divide by negative values
# (unless the dataset is really dodgy!):
# cython: cdivision=True

"""
Interpolation to determine wind velocity at any given time,
latitude, longitude and altitude.

Note that this module is compiled with Cython to enable fast
memory access.
"""


from magicmemoryview import MagicMemoryView
from .warnings cimport WarningCounts
import numpy as np
from .dataset import Dataset
from math import exp, log

cimport numpy as np

# These need to match Dataset.axes.variable
DEF VAR_A = 0
DEF VAR_U = 1
DEF VAR_V = 2
DEF VAR_T = 3


ctypedef float[:, :, :, :, :] dataset_t # TODO rename

cdef struct Lerp1:
    long index
    double interpolation_weight

cdef struct Lerp3:
    long hour, latitude_index, longitude_index
    double interpolation_weight


class RangeError(ValueError):
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
        s = "{0}={1}".format(variable, value)
        super(RangeError, self).__init__(s)


def make_interpolator(dataset, WarningCounts warnings):
    """
    Produce a function that can get wind data from `dataset`

    This wrapper casts :attr:`Dataset.array` into a form that is useful
    to us, and then returns a closure that can be used to retrieve
    wind velocities.
    """

    cdef float[:, :, :, :, :] data

    if warnings is None:
        raise TypeError("Warnings must not be None")

    data = MagicMemoryView(dataset.array, Dataset.shape, b"f")

    def f(hour, lat, lng, alt):
        return get_atmospheric_state(data, warnings, hour, lat, lng, alt)

    return f


cdef object get_atmospheric_state(dataset_t ds,
                                 WarningCounts warnings, double hour,
                                 double lat, double lng, double alt):
    """
    Interpolates the [u, v] wind components, temperature, and pressure at the given time and location.
    
    :param ds: The float array representing the dataset.
    :param warnings: The warning accumulator
    :param hour: The hour at which to get the atmospheric data, in fractional hours since the dataset start
    :param lat: The latitude at which to get the atmospheric data, in decimal degrees (-90 to 90)
    :param lng: The longitude at which to get the atmospheric data, in decimal degrees (0 to 360)
    :param alt: The altitude at which to get the atmospheric data, in meters above sea level
    :return: The interpolated [u, v] wind components, in meters/second, temperature, in Kelvin, and pressure, in millibar
    """

    cdef Lerp3[8] lerps
    cdef long altitude_index
    cdef double lower, upper, u, v

    pick3(hour, lat, lng, lerps)

    altitude_index = find_altitude_index(ds, lerps, alt)
    lower = interpolate_lat_lng_time(ds, lerps, VAR_A, altitude_index)
    upper = interpolate_lat_lng_time(ds, lerps, VAR_A, altitude_index + 1)

    if lower != upper:
        interpolation_weight = (upper - alt) / (upper - lower)
    else:
        interpolation_weight = 0.5

    if interpolation_weight < 0: warnings.altitude_too_high += 1

    cdef Lerp1 alt_lerp = Lerp1(altitude_index, interpolation_weight)

    u = interp4(ds, lerps, alt_lerp, VAR_U)
    v = interp4(ds, lerps, alt_lerp, VAR_V)
    t = interp4(ds, lerps, alt_lerp, VAR_T)

    p = interp_exponential(Dataset.pressures_sorted[altitude_index],
                           Dataset.pressures_sorted[altitude_index + 1],
                           interpolation_weight)

    return u, v, t, p

cdef long pick(double left, double step_size, long num_steps, double value,
               object variable_name, Lerp1[2] out) except -1:
    """ 
    TODO 
    :param left:  
    :param step_size: The size of steps in the set from which to interpolate. 
    :param num_steps: The number of steps in the set from which to interpolate. 
    :param value: The value at which to interpolate. 
    :param variable_name: The name of the variable. 
    :param out:  
    :return:  
    """

    cdef double a, l
    cdef long b

    a = (value - left) / step_size
    b = <long> a
    if b < 0 or b >= num_steps - 1:
        raise RangeError(variable_name, value)
    l = a - b # discard integer part (characteristic)

    out[0] = Lerp1(b, 1 - l)
    out[1] = Lerp1(b + 1, l)
    return 0

cdef long pick3(double hour, double lat, double lng, Lerp3[8] out) except -1:
    """ 
    TODO 
    :param hour:  
    :param lat:  
    :param lng:  
    :param out:  
    :return:  
    """
    cdef Lerp1[2] lhour, llat, llng

    # the dimensions of the lat/lon axes are 361 and 720
    # (The latitude axis includes its two endpoints; the longitude only
    # includes the lower endpoint)
    # However, the longitude does wrap around, so we tell `pick` that the
    # longitude axis is one larger than it is (so that it can "choose" the
    # 721st point/the 360 degrees point), then wrap it afterwards.
    pick(0, 3, 65, hour, "hour", lhour)
    pick(-90, 0.5, 361, lat, "lat", llat)
    pick(0, 0.5, 720 + 1, lng, "lng", llng)
    if llng[1].index == 720:
        llng[1].index = 0

    cdef long i = 0

    for a in lhour:
        for b in llat:
            for c in llng:
                p = a.interpolation_weight * b.interpolation_weight * c.interpolation_weight
                out[i] = Lerp3(a.index, b.index, c.index, p)
                i += 1

    return 0

cdef double interpolate_lat_lng_time(dataset_t ds,
                                     Lerp3[8] lerps, long variable,
                                     long level_index):
    """ 
    Interpolate a variable at a given pressure level with respect to latitude, 
    longitude, and time. 
    :param dataset_t ds: The dataset to interpolate from.  
    :param lerps: TODO 
    :param long variable: The dataset index of the variable to interpolate. 
    :param long level_index: The index of the pressure level to interpolate at. 
    :return: The value of the interpolated variable. 
    """
    cdef double interpolated_value, dataset_value

    interpolated_value = 0
    for i in range(8):
        lerp = lerps[i]
        dataset_value = ds[lerp.hour, level_index, variable,
                           lerp.latitude_index, lerp.longitude_index]
        # Interpolation value is weighted average of nearest dataset values
        interpolated_value += \
            dataset_value * lerp.interpolation_weight

    return interpolated_value


# Searches for the largest index lower than target, excluding the topmost level.
cdef long find_altitude_index(dataset_t ds, Lerp3[8] lerps, double target):
    """ 
    Search for the largest altitude index lower than the target, excluding the topmost level. 
    :param dataset_t ds: The dataset to search. 
    :param lerps: TODO 
    :param target: The altitude to search for 
    :return: The index of the highest altitude layer lower than the target that. 
    """

    # Searches for the largest index lower than target, excluding the topmost level.
    cdef long lower, upper, mid
    cdef double test
    
    lower, upper = 0, 45

    while lower < upper: # Search by bisection
        mid = (lower + upper + 1) / 2
        test = interpolate_lat_lng_time(ds, lerps, VAR_A, mid)
        if target <= test:
            upper = mid - 1
        else:
            lower = mid

    return lower

cdef double interp4(dataset_t ds,
                    Lerp3[8] lerps,
                    Lerp1 alt_lerp,
                    long variable):
    lower = interpolate_lat_lng_time(ds, lerps, variable, alt_lerp.index)
    # and we can infer what the other lerp1 is...
    upper = interpolate_lat_lng_time(ds, lerps, variable, alt_lerp.index + 1)
    return lower * alt_lerp.interpolation_weight + upper * (1 - alt_lerp.interpolation_weight)

cdef double interp_exponential(double lower, double upper,
                            double interpolation_weight):
    """
    Interpolates between two values based on an exponential curve fit between them
    :param lower: The lower of the values to interpolate between
    :param upper: The upper of the values to interpolate between
    :param interpolation_weight: The interpolation weight
    :return: The interpolated value
    """
    return lower*exp(log(upper/lower)*interpolation_weight)  # TODO this is questionable
