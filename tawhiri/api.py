# Copyright 2014 (C) Priyesh Patel
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
Provide the HTTP API for Tawhiri.
"""

from flask import Flask, jsonify, request, g
from datetime import datetime
import time
import strict_rfc3339

from tawhiri import solver, models, warnings
from tawhiri.dataset import Dataset as WindDataset
from ruaumoko import Dataset as ElevationDataset

app = Flask(__name__)

API_VERSION = "1.2"
LATEST_DATASET_KEYWORD = "latest"
PROFILE_STANDARD = "standard_profile"
PROFILE_FLOAT = "float_profile"
PHYSICS_MODEL_CUSF = "CUSF"
PHYSICS_MODEL_BPP = "UMDBPP"
PHYSICS_MODEL_BPP_DIAMETER_TERM = "UMDBPP-B"

DEFAULT_WIND_AZ_STD_DEV = 5*3.14159265358979232/180  # TODO


# Util functions ##############################################################
def ruaumoko_ds():
    if not hasattr("ruaumoko_ds", "once"):
        ds_loc = app.config.get('ELEVATION_DATASET', ElevationDataset.default_location)
        ruaumoko_ds.once = ElevationDataset(ds_loc)

    return ruaumoko_ds.once


def _rfc3339_to_timestamp(dt):
    """
    Convert from a RFC3339 timestamp to a UNIX timestamp.
    """
    return strict_rfc3339.rfc3339_to_timestamp(dt)


def _timestamp_to_rfc3339(dt):
    """
    Convert from a UNIX timestamp to a RFC3339 timestamp.
    """
    return strict_rfc3339.timestamp_to_rfc3339_utcoffset(dt)


# Exceptions ##################################################################
class APIException(Exception):
    """
    Base API exception.
    """
    status_code = 500


class RequestException(APIException):
    """
    Raised if request is invalid.
    """
    status_code = 400


class InvalidDatasetException(APIException):
    """
    Raised if the dataset specified in the request is invalid.
    """
    status_code = 404


class PredictionException(APIException):
    """
    Raised if the solver raises an exception.
    """
    status_code = 500


class InternalException(APIException):
    """
    Raised when an internal error occurs.
    """
    status_code = 500


class NotYetImplementedException(APIException):
    """
    Raised when the functionality has not yet been implemented.
    """
    status_code = 501


# Request #####################################################################
def parse_request(data):
    """
    Parse the request.
    """
    request = {"version": API_VERSION}

    # Generic fields
    request['launch_latitude'] = \
        _extract_parameter(data, "launch_latitude", float,
                           validator=lambda x: -90 <= x <= 90)
    request['launch_longitude'] = \
        _extract_parameter(data, "launch_longitude", float,
                           validator=lambda x: 0 <= x < 360)
    request['launch_datetime'] = \
        _extract_parameter(data, "launch_datetime", _rfc3339_to_timestamp)
    request['launch_altitude'] = \
        _extract_parameter(data, "launch_altitude", float, ignore=True)

    # If no launch altitude provided, use Ruaumoko to look it up
    if request['launch_altitude'] is None:
        try:
            request['launch_altitude'] = ruaumoko_ds().get(request['launch_latitude'],
                                                       request['launch_longitude'])
        except Exception:
            raise InternalException("Internal exception experienced whilst " +
                                    "looking up 'launch_altitude'.")

    launch_alt = request["launch_altitude"]

    # Prediction profile
    request['profile'] = _extract_parameter(data, "profile", str,
                                            PROFILE_STANDARD)

    request['physics_model'] = _extract_parameter(data, "physics_model", str,
                                                  default=PHYSICS_MODEL_CUSF)

    request['monte_carlo'] = _extract_parameter(data, "monte_carlo", bool,
                                                default=False)

    if request['profile'] == PROFILE_STANDARD:

        request['descent_rate'] = \
            _extract_parameter(data, "descent_rate", float,
                               validator=lambda x: x > 0)

        if request['monte_carlo']:
            request['descent_rate_std_dev'] = \
                _extract_parameter(data, "descent_rate_std_dev", float,
                                   default=0,
                                   validator=lambda x: x >= 0)

            request['wind_mag_std_dev'] = \
                _extract_parameter(data, "wind_std_dev", float,
                                   default=0,
                                   validator=lambda x: x >= 0)
            if request['wind_mag_std_dev'] != 0:
                request['wind_azimuth_std_dev'] = DEFAULT_WIND_AZ_STD_DEV
            else:
                request['wind_azimuth_std_dev'] = 0

        else:
            request['descent_rate_std_dev'] = 0
            request['wind_mag_std_dev'] = 0
            request['wind_azimuth_std_dev'] = 0

        if request['physics_model'] == PHYSICS_MODEL_CUSF:
            request['burst_altitude'] = \
                _extract_parameter(data, "burst_altitude", float,
                                   validator=lambda x: x > launch_alt)

            request['ascent_rate'] = \
                _extract_parameter(data, "ascent_rate", float,
                                   validator=lambda x: x > 0)

            if request['monte_carlo']:
                request['burst_altitude_std_dev'] = \
                    _extract_parameter(data, "burst_altitude_std_dev", float,
                                       default=0,
                                       validator=lambda x: x > launch_alt)

                request['ascent_rate_std_dev'] = \
                    _extract_parameter(data, "ascent_rate_std_dev", float,
                                       default=0,
                                       validator=lambda x: x >= 0)

            else:
                request['burst_altitude_std_dev'] = 0
                request['ascent_rate_std_dev'] = 0

        elif request['physics_model'] == PHYSICS_MODEL_BPP:
            request['burst_altitude'] = \
                _extract_parameter(data, "burst_altitude", float,
                                   validator=lambda x: x > launch_alt)

            request['helium_mass'] = \
                _extract_parameter(data, "helium_mass", float,
                                   validator=lambda x: x >= 0)

            request['payload_mass'] = \
                _extract_parameter(data, "payload_mass", float,
                                   validator=lambda x: x >= 0)

            request['balloon_mass'] = \
                _extract_parameter(data, "balloon_mass", float,
                                   validator=lambda x: x >= 0)

            if request['monte_carlo']:
                request['burst_altitude_std_dev'] = \
                    _extract_parameter(data, "burst_altitude_std_dev", float,
                                       default=0,
                                       validator=lambda x: x > launch_alt)

                request['helium_mass_std_dev'] = \
                    _extract_parameter(data, "helium_mass_std_dev", float,
                                       default=0,
                                       validator=lambda x: x >= 0)

            else:
                request['burst_altitude_std_dev'] = 0
                request['helium_mass_std_dev'] = 0

        elif request['physics_model'] == PHYSICS_MODEL_BPP_DIAMETER_TERM:
            request['burst_diameter'] = \
                _extract_parameter(data, "burst_diameter", float,
                                   validator=lambda x: x > 0)

            request['helium_mass'] = \
                _extract_parameter(data, "helium_mass", float,
                                   validator=lambda x: x >= 0)

            request['payload_mass'] = \
                _extract_parameter(data, "payload_mass", float,
                                   validator=lambda x: x >= 0)

            request['balloon_mass'] = \
                _extract_parameter(data, "balloon_mass", float,
                                   validator=lambda x: x >= 0)

            if request['monte_carlo']:
                request['burst_diameter_std_dev'] = \
                    _extract_parameter(data, "burst_diameter_std_dev", float,
                                       default=0,
                                       validator=lambda x: x > 0)

                request['helium_mass_std_dev'] = \
                    _extract_parameter(data, "helium_mass_std_dev", float,
                                       default=0,
                                       validator=lambda x: x >= 0)

            else:
                request['burst_diameter_std_dev'] = 0
                request['helium_mass_std_dev'] = 0

        else:
            raise RequestException(
                "Unknown physics model '%s'." % request['physics_model'])

    elif request['profile'] == PROFILE_FLOAT:
        request['ascent_rate'] = _extract_parameter(data, "ascent_rate", float,
                                                validator=lambda x: x > 0)
        request['float_altitude'] = \
            _extract_parameter(data, "float_altitude", float,
                               validator=lambda x: x > launch_alt)
        request['stop_datetime'] = \
            _extract_parameter(data, "stop_datetime", _rfc3339_to_timestamp,
                               validator=lambda x: x > request['launch_datetime'])
    else:
        raise RequestException("Unknown profile '%s'." % request['profile'])

    # Dataset
    request['dataset'] = _extract_parameter(data, "dataset", _rfc3339_to_timestamp,
                                            default=LATEST_DATASET_KEYWORD)

    return request


def _extract_parameter(data, parameter, cast, default=None, ignore=False,
                       validator=None):
    """
    Extract a parameter from the POST request and raise an exception if any
    parameter is missing or invalid.
    """
    if parameter not in data:
        if default is None and not ignore:
            raise RequestException("Parameter '%s' not provided in request." %
                                   parameter)
        return default

    try:
        result = cast(data[parameter])
    except Exception:
        raise RequestException("Unable to parse parameter '%s': %s." %
                               (parameter, data[parameter]))

    if validator is not None and not validator(result):
        raise RequestException("Invalid value for parameter '%s': %s." %
                               (parameter, data[parameter]))

    return result


# Response ####################################################################
def run_prediction(req):
    """
    Run the prediction.
    """
    # Response dict
    resp = {
        "request": req,
        "prediction": [],
    }

    # Find wind data location
    ds_dir = app.config.get('WIND_DATASET_DIR', WindDataset.DEFAULT_DIRECTORY)

    # Dataset
    try:
        if req['dataset'] == LATEST_DATASET_KEYWORD:
            tawhiri_ds = WindDataset.open_latest(persistent=True, directory=ds_dir)
        else:
            tawhiri_ds = WindDataset(datetime.fromtimestamp(req['dataset']), directory=ds_dir)
    except IOError:
        raise InvalidDatasetException("No matching dataset found.")
    except ValueError as e:
        raise InvalidDatasetException(*e.args)

    # Note that hours and minutes are set to 00 as Tawhiri uses hourly datasets
    resp['request']['dataset'] = tawhiri_ds.ds_time.strftime(
        "%Y-%m-%dT%H:00:00Z")

    warningcounts = warnings.WarningCounts()

    # Stages
    if req['profile'] == PROFILE_STANDARD:

        if req['physics_model'] == PHYSICS_MODEL_CUSF:
            stages = models.standard_profile_cusf(req['ascent_rate'],
                                                  req['burst_altitude'],
                                                  req['descent_rate'], tawhiri_ds,
                                                  ruaumoko_ds(), warningcounts,
                                                  req['ascent_rate_std_dev'],
                                                  req['burst_altitude_std_dev'],
                                                  req['descent_rate_std_dev'],
                                                  req['wind_mag_std_dev'],
                                                  req['wind_azimuth_std_dev'])
        elif req['physics_model'] == PHYSICS_MODEL_BPP:
            stages = models.standard_profile_bpp(req['helium_mass'],
                                                 req['payload_mass'] + req['balloon_mass'],
                                                 req['burst_altitude'],
                                                 req['descent_rate'], tawhiri_ds,
                                                 ruaumoko_ds(), warningcounts,
                                                 req['burst_altitude_std_dev'],
                                                 req['descent_rate_std_dev'],
                                                 req['wind_mag_std_dev'],
                                                 req['wind_azimuth_std_dev'],
                                                 req['helium_mass_std_dev'])
        elif req['physics_model'] == PHYSICS_MODEL_BPP_DIAMETER_TERM:
            stages = models.standard_profile_bpp_diameter_term(
                req['helium_mass'], req['payload_mass'] + req['balloon_mass'],
                req['burst_diameter'], req['descent_rate'], tawhiri_ds,
                ruaumoko_ds(), warningcounts, req['burst_diameter_std_dev'],
                req['descent_rate_std_dev'], req['wind_mag_std_dev'],
                req['wind_azimuth_std_dev'], req['helium_mass_std_dev'])

        else:
            raise InternalException("Unknown physics model '%s'." % req['physics_model'])


    elif req['profile'] == PROFILE_FLOAT:
        stages = models.float_profile(req['ascent_rate'],
                                      req['float_altitude'],
                                      req['stop_datetime'], tawhiri_ds,
                                      warningcounts)
    else:
        raise InternalException("No implementation for known profile.")

    # Run solver
    try:
        result = solver.solve(req['launch_datetime'], req['launch_latitude'],
                              req['launch_longitude'], req['launch_altitude'],
                              stages)
    except Exception as e:
        raise PredictionException("Prediction did not complete: '%s'." %
                                  str(e))

    # Format trajectory
    if req['profile'] == PROFILE_STANDARD:
        resp['prediction'] = _parse_stages(["ascent", "descent"], result)
    elif req['profile'] == PROFILE_FLOAT:
        resp['prediction'] = _parse_stages(["ascent", "float"], result)
    else:
        raise InternalException("No implementation for known profile.")

    # Convert request UNIX timestamps to RFC3339 timestamps
    for key in resp['request']:
        if "datetime" in key:
            resp['request'][key] = _timestamp_to_rfc3339(resp['request'][key])

    return resp


def _parse_stages(labels, data):
    """
    Parse the predictor output for a set of stages.
    """
    assert len(labels) == len(data)

    prediction = []
    for index, leg in enumerate(data):
        stage = {}
        stage['stage'] = labels[index]
        stage['trajectory'] = [{
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'datetime': _timestamp_to_rfc3339(dt),
            } for dt, lat, lon, alt in leg]
        prediction.append(stage)
    return prediction


# Flask App ###################################################################
@app.route('/api/v{0}/'.format(API_VERSION), methods=['GET'])
def main():
    """
    Single API endpoint which accepts GET requests.
    """
    g.request_start_time = time.time()
    response = run_prediction(parse_request(request.args))
    g.request_complete_time = time.time()
    response['metadata'] = _format_request_metadata()
    return jsonify(response)


@app.errorhandler(APIException)
def handle_exception(error):
    """
    Return correct error message and HTTP status code for API exceptions.
    """
    response = {}
    response['error'] = {
        "type": type(error).__name__,
        "description": str(error)
    }
    g.request_complete_time = time.time()
    response['metadata'] = _format_request_metadata()
    return jsonify(response), error.status_code


def _format_request_metadata():
    """
    Format the request metadata for inclusion in the response.
    """
    return {
        "start_datetime": _timestamp_to_rfc3339(g.request_start_time),
        "complete_datetime": _timestamp_to_rfc3339(g.request_complete_time),
    }
