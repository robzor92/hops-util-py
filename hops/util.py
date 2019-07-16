"""

Miscellaneous utility functions for user applications.

"""

import os
import signal
from ctypes import cdll
import itertools
import socket
import json
import ssl
import jks
from pathlib import Path
import fnmatch

from hops import hdfs
from hops import constants
from hops import tls
from hops import devices

#! Needed for hops library backwards compatability
try:
    import requests
except:
    pass

import pydoop.hdfs

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

    print(data)

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

def _find_in_path(path, file):
    """
    Utility method for finding a filename-string in a path

    Args:
        :path: the path to search
        :file: the filename to search for

    Returns:
        True if the filename was found in the path, otherwise False

    """
    for p in path.split(os.pathsep):
        candidate = os.path.join(p, file)
        if (os.path.exists(os.path.join(p, file))):
            return candidate
    return False

def _find_tensorboard():
    """
    Utility method for finding the tensorboard binary

    Returns:
         tb_path, path to the binary
    """
    pypath = os.getenv("PYSPARK_PYTHON")
    pydir = os.path.dirname(pypath)
    search_path = os.pathsep.join([pydir, os.environ[constants.ENV_VARIABLES.PATH_ENV_VAR], os.environ[constants.ENV_VARIABLES.PYTHONPATH_ENV_VAR]])
    tb_path = _find_in_path(search_path, 'tensorboard')
    if not tb_path:
        raise Exception("Unable to find 'tensorboard' in: {}".format(search_path))
    return tb_path

def _on_executor_exit(signame):
    """
    Return a function to be run in a child process which will trigger
    SIGNAME to be sent when the parent process dies

    Args:
        :signame: the signame to send

    Returns:
        set_parent_exit_signal
    """
    signum = getattr(signal, signame)
    def set_parent_exit_signal():
        # http://linux.die.net/man/2/prctl

        PR_SET_PDEATHSIG = 1
        result = cdll['libc.so.6'].prctl(PR_SET_PDEATHSIG, signum)
        if result != 0:
            raise Exception('prctl failed with error code %s' % result)
    return set_parent_exit_signal

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


def num_executors():
    """
    Get the number of executors configured for Jupyter

    Returns:
        Number of configured executors for Jupyter
    """
    sc = _find_spark().sparkContext
    return int(sc._conf.get("spark.dynamicAllocation.maxExecutors"))

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

def grid_params(dict):
    """
    Generate all possible combinations (cartesian product) of the hyperparameter values

    Args:
        :dict:

    Returns:
        A new dictionary with a grid of all the possible hyperparameter combinations
    """
    keys = dict.keys()
    val_arr = []
    for key in keys:
        val_arr.append(dict[key])

    permutations = list(itertools.product(*val_arr))

    args_dict = {}
    slice_index = 0
    for key in keys:
        args_arr = []
        for val in list(zip(*permutations))[slice_index]:
            args_arr.append(val)
        slice_index += 1
        args_dict[key] = args_arr
    return args_dict

def _get_ip_address():
    """
    Simple utility to get host IP address

    Returns:
        ip address of current host
    """
    try:
        _, _, _, _, addr = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)[0]
        return addr[0]
    except:
        return socket.gethostbyname(socket.getfqdn())

def _time_diff(task_start, task_end):
    """
    Args:
        :task_start: time in microseconds
        :tast_end: time in microseconds

    Returns:

    """

    task_start = _microseconds_to_millis(task_start)
    task_end = _microseconds_to_millis(task_end)

    millis = task_end - task_start

    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24

    return "%d hours, %d minutes, %d seconds" % (hours, minutes, seconds)

def _microseconds_to_millis(time):
    return int(round(time * 1000))

def _publish_experiment(appid, elastic_id, json_data, xattr):
    """
    Utility method for putting JSON data into elastic search

    Args:
        :project: the project of the user/app
        :appid: the YARN appid
        :elastic_id: the id in elastic
        :json_data: the data to put

    Returns:
        None

    """
    headers = {'Content-type': 'application/json'}
    resource_url = constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_REST_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_PROJECT_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   hdfs.project_id() + constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_EXPERIMENTS_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   appid + "_" + str(elastic_id) + "?xattr=" + xattr

    resp = send_request_with_session('POST', resource_url, data=json_data, headers=headers)
    print(resp)




def _populate_experiment(sc, model_name, module, function, hyperparameter_space, versioned_resources, description):
    """
    Args:
         :sc:
         :model_name:
         :module:
         :function:
         :logdir:
         :hyperparameter_space:
         :versioned_resources:
         :description:

    Returns:

    """
    return json.dumps({'name': model_name, 'description': description, 'state': 'RUNNING'})

def _finalize_experiment(experiment_json, hyperparameter, metric, state, duration, hp_combs):
    """
    Args:
        :experiment_json:
        :hyperparameter:
        :metric:

    Returns:

    """
    experiment_json = json.loads(experiment_json)
    experiment_json['metric'] = metric
    experiment_json['state'] = state
    experiment_json['duration'] = duration
    experiment_json['results'] = hp_combs

    #experiment_json['hyperparameter'] = hyperparameter

    return json.dumps(experiment_json)

def _store_local_tensorboard(local_tb, hdfs_exec_logdir):
    """

    Args:
        :local_tb:
        :hdfs_exec_logdir:

    Returns:

    """
    tb_contents = os.listdir(local_tb)
    for entry in tb_contents:
        pydoop.hdfs.put(local_tb + '/' + entry, hdfs_exec_logdir)

def _version_resources(versioned_resources, rundir):
    """

    Args:
        versioned_resources:
        rundir:

    Returns:

    """
    if not versioned_resources:
        return None
    pyhdfs_handle = hdfs.get()
    pyhdfs_handle.create_directory(rundir)
    endpoint_prefix = hdfs.project_path()
    versioned_paths = []
    for hdfs_resource in versioned_resources:
        if pydoop.hdfs.path.exists(hdfs_resource):
            pyhdfs_handle.copy(hdfs_resource, pyhdfs_handle, rundir)
            path, filename = os.path.split(hdfs_resource)
            versioned_paths.append(rundir.replace(endpoint_prefix, '') + '/' + filename)
        else:
            raise Exception('Could not find resource in specified path: ' + hdfs_resource)

    return ', '.join(versioned_paths)

def _convert_to_dict(best_param):
    """
    Utiliy method for converting best_param string to dict

    Args:
        :best_param: the best_param string

    Returns:
        a dict with param->value

    """


    best_param_dict={}
    best_param = best_param.split('&')
    for hp in best_param:
        hp = hp.split('=')
        best_param_dict[hp[0]] = hp[1]

    return best_param_dict

def _convert_param_to_arr(best_param):
    best_param_arr=[]
    best_param = best_param.split('&')
    for hp in best_param:
        hp = hp.split('=')
        best_param_arr.append({hp[0]: hp[1]})

    return best_param_arr

def _find_spark():
    """

    Returns: SparkSession

    """
    return SparkSession.builder.getOrCreate()

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

def _get_experiments_dir():
    """
    Gets the root folder where the experiments are writing their results

    Returns:
        The folder where the experiments are writing results
    """
    pyhdfs_handle = hdfs.get()
    assert pyhdfs_handle.exists(hdfs.project_path() + "Experiments"), "Your project is missing an Experiments dataset, please create one."
    return hdfs.project_path() + "Experiments"

def _get_logdir(app_id, run_id):
    """

    Args:
        app_id: app_id for experiment
        run_id: run_id for experiment

    Returns:
        The folder where a particular experiment is writing results

    """
    return _get_experiments_dir() + '/' + str(app_id) + '_' + str(run_id)

def _create_experiment_subdirectories(app_id, run_id, param_string, type, sub_type=None):
    """
    Creates directories for an experiment, if Experiments folder exists it will create directories
    below it, otherwise it will create them in the Logs directory.

    Args:
        :app_id: YARN application ID of the experiment
        :run_id: Experiment ID
        :param_string: name of the new directory created under parent directories
        :type: type of the new directory parent, e.g differential_evolution
        :sub_type: type of sub directory to parent, e.g generation

    Returns:
        The new directories for the yarn-application and for the execution (hdfs_exec_logdir, hdfs_appid_logdir)
    """

    pyhdfs_handle = hdfs.get()

    hdfs_events_parent_dir = hdfs.project_path() + "Experiments"

    hdfs_experiment_dir = hdfs_events_parent_dir + "/" + app_id + "_" + str(run_id)

    # determine directory structure based on arguments
    if sub_type:
        hdfs_exec_logdir = hdfs_experiment_dir + "/" + str(sub_type) + '/' + str(param_string)
    elif not param_string and not sub_type:
        hdfs_exec_logdir = hdfs_experiment_dir + '/'
    else:
        hdfs_exec_logdir = hdfs_experiment_dir + '/' + str(param_string)

    # Need to remove directory if it exists (might be a task retry)
    if pyhdfs_handle.exists(hdfs_exec_logdir):
        hdfs.delete(hdfs_exec_logdir, recursive=True)

    # create the new directory
    pyhdfs_handle.create_directory(hdfs_exec_logdir)

    # update logfile
    logfile = hdfs_exec_logdir + '/' + 'logfile'
    os.environ['EXEC_LOGFILE'] = logfile

    return hdfs_exec_logdir, hdfs_experiment_dir


def _cleanup(local_logdir_bool, local_tb_path, hdfs_exec_logdir, gpu_thread, tb_hdfs_file):
    try:
        if local_logdir_bool:
            _store_local_tensorboard(local_tb_path, hdfs_exec_logdir)
    except Exception as err:
        print('Exception occurred while uploading local logdir to hdfs: {}'.format(err))
    finally:
        if devices.get_num_gpus() > 0 and gpu_thread.isAlive():
            gpu_thread.do_run = False
            gpu_thread.join(20)

        handle = hdfs.get()
        try:
            if tb_hdfs_file and handle.exists(tb_hdfs_file):
                handle.delete(tb_hdfs_file)
        except:
            pass

# Search for .metric file in max two levels
def _build_hyperparameter_json(logdir):

    hyperparameters = []
    metric_files = []

    for experiment_dir in hdfs.ls(logdir):
        runs = hdfs.ls(experiment_dir, recursive=True)
        for run in runs:
            if run.endswith('.metric'):
                metric_files.append(run)

    for metric_file in metric_files:
        metric_file = hdfs.abs_path(metric_file)
        hyperparameter_combination = os.path.split(os.path.dirname(metric_file))[1]
        hp_arr = _convert_param_to_arr(hyperparameter_combination)

        metric = hdfs.load(metric_file).decode("UTF-8")
        hyperparameters.append({'metrics': [{'metric': metric}], 'hyperparameters': hp_arr})

    return hyperparameters






