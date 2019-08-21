"""

Miscellaneous utility functions for user applications.

"""

import os
import ssl
import jks
from pathlib import Path
from hops import constants
from hops import tls
#! Needed for hops library backwards compatability
try:
    import requests
except:
    pass

try:
    import tensorflow
except:
    pass

try:
    import http.client as http
except ImportError:
    import httplib as http

# in case importing in %%local
try:
    from pyspark.sql import SparkSession
except:
    pass

def _convert_jks_to_pem():
    ca = jks.KeyStore.load("domain_ca_truststore", "adminpw", try_decrypt_keys=True)
    ca_certs = ""
    for alias, c in ca.certs.items():
        ca_certs = ca_certs + tls._bytes_to_pem_str(c.cert, "CERTIFICATE")
    ca_cert_path = Path("catrust.pem")
    with ca_cert_path.open("w") as f:
        f.write(ca_certs)

def send_request_with_session(method, resource, data=None, headers=None):
    """
    Sends a request to Hopsworks over HTTPS. In case of Unauthorized response, submit the request once more as jwt
    might not have been read properly from local container.

    Args:
        method: request method
        url: Hopsworks request url
        data: request data payload
        headers: request headers

    Returns:
        HTTPS response
    """
    if not os.path.exists("catrust.pem"):
        _convert_jks_to_pem()

    if headers is None:
        headers = {}
    headers[constants.HTTP_CONFIG.HTTP_AUTHORIZATION] = "Bearer " + get_jwt()
    session = requests.session()
    host_port_pair = _get_host_port_pair()
    url = "https://" + host_port_pair[0] + ":" + host_port_pair[1] + resource
    req = requests.Request(method, url, data=data, headers=headers)
    prepped = session.prepare_request(req)
    response = session.send(prepped, verify="catrust.pem")

    if response.status_code == constants.HTTP_CONFIG.HTTP_UNAUTHORIZED:
        req.headers[constants.HTTP_CONFIG.HTTP_AUTHORIZATION] = "Bearer " + get_jwt()
        prepped = session.prepare_request(req)
        response = session.send(prepped)
    return response

def _get_elastic_endpoint():
    """

    Returns:
        The endpoint for putting things into elastic search

    """
    elastic_endpoint = os.environ[constants.ENV_VARIABLES.ELASTIC_ENDPOINT_ENV_VAR]
    host, port = elastic_endpoint.split(':')
    return host + ':' + port

def _get_hopsworks_rest_endpoint():
    """

    Returns:
        The hopsworks REST endpoint for making requests to the REST API

    """
    return os.environ[constants.ENV_VARIABLES.REST_ENDPOINT_END_VAR]

def _get_host_port_pair():
    """
    Removes "http or https" from the rest endpoint and returns a list
    [endpoint, port], where endpoint is on the format /path.. without http://

    Returns:
        a list [endpoint, port]
    """
    endpoint = _get_hopsworks_rest_endpoint()
    if 'http' in endpoint:
        last_index = endpoint.rfind('/')
        endpoint = endpoint[last_index + 1:]
    host_port_pair = endpoint.split(':')
    return host_port_pair

def _get_http_connection(https=False):
    """
    Opens a HTTP(S) connection to Hopsworks

    Args:
        https: boolean flag whether to use Secure HTTP or regular HTTP

    Returns:
        HTTP(S)Connection
    """
    host_port_pair = _get_host_port_pair()
    if (https):
        PROTOCOL = ssl.PROTOCOL_TLSv1_2
        ssl_context = ssl.SSLContext(PROTOCOL)
        connection = http.HTTPSConnection(str(host_port_pair[0]), int(host_port_pair[1]), context = ssl_context)
    else:
        connection = http.HTTPConnection(str(host_port_pair[0]), int(host_port_pair[1]))
    return connection


def send_request(connection, method, resource, body=None, headers=None):
    """
    Sends a request to Hopsworks. In case of Unauthorized response, submit the request once more as jwt might not
    have been read properly from local container.

    Args:
        connection: HTTP connection instance to Hopsworks
        method: HTTP(S) method
        resource: Hopsworks resource
        body: HTTP(S) body
        headers: HTTP(S) headers

    Returns:
        HTTP(S) response
    """
    if headers is None:
        headers = {}
    headers[constants.HTTP_CONFIG.HTTP_AUTHORIZATION] = "Bearer " + get_jwt()
    connection.request(method, resource, body, headers)
    response = connection.getresponse()
    if response.status == constants.HTTP_CONFIG.HTTP_UNAUTHORIZED:
        headers[constants.HTTP_CONFIG.HTTP_AUTHORIZATION] = "Bearer " + get_jwt()
        connection.request(method, resource, body, headers)
        response = connection.getresponse()
    return response

def _parse_rest_error(response_dict):
    """
    Parses a JSON response from hopsworks after an unsuccessful request

    Args:
        response_dict: the JSON response represented as a dict

    Returns:
        error_code, error_msg, user_msg
    """
    error_code = -1
    error_msg = ""
    user_msg = ""
    if constants.REST_CONFIG.JSON_ERROR_CODE in response_dict:
        error_code = response_dict[constants.REST_CONFIG.JSON_ERROR_CODE]
    if constants.REST_CONFIG.JSON_ERROR_MSG in response_dict:
        error_msg = response_dict[constants.REST_CONFIG.JSON_ERROR_MSG]
    if constants.REST_CONFIG.JSON_USR_MSG in response_dict:
        user_msg = response_dict[constants.REST_CONFIG.JSON_USR_MSG]
    return error_code, error_msg, user_msg

def get_job_name():
    """
    If this method is called from inside a hopsworks job, it returns the name of the job.

    Returns:
        the name of the hopsworks job

    """
    if constants.ENV_VARIABLES.JOB_NAME_ENV_VAR in os.environ:
        return os.environ[constants.ENV_VARIABLES.JOB_NAME_ENV_VAR]
    else:
        None

def get_jwt():
    """
    Retrieves jwt from local container

    Returns:
        Content of jwt.token file in local container.
    """
    with open(constants.REST_CONFIG.JWT_TOKEN, "r") as jwt:
        return jwt.read()

def num_executors():
    """
    Get the number of executors configured for Jupyter

    Returns:
        Number of configured executors for Jupyter
    """
    sc = _find_spark().sparkContext
    try:
        return int(sc._conf.get("spark.dynamicAllocation.maxExecutors"))
    except:
        raise RuntimeError('Failed to find spark.dynamicAllocation.maxExecutors property, please select your mode as either Experiment, Parallel Experiments or Distributed Training.')

def num_param_servers():
    """
    Get the number of parameter servers configured for Jupyter

    Returns:
        Number of configured parameter servers for Jupyter
    """
    sc = _find_spark().sparkContext
    try:
        return int(sc._conf.get("spark.tensorflow.num.ps"))
    except:
        return 0

def _find_spark():
    """
    Returns: SparkSession
    """
    return SparkSession.builder.getOrCreate()




