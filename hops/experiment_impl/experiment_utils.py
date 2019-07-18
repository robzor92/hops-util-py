"""
Utility functions for experiments
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

from hops import hdfs

def _handle_return(retval, hdfs_exec_logdir, optimization_key):
    """

    Args:
        val:
        hdfs_exec_logdir:

    Returns:

    """
    # Validation
    if not optimization_key and type(retval) is dict and len(retval.keys()) > 1:
        raise Exception('Missing optimization_key argument, when returning multiple values in a dict the optimization_key argument must be set.')
    elif type(retval) is dict and optimization_key not in retval:
        raise Exception('Specified optimization key {} not in returned dict'.format(optimization_key))
    elif type(retval) is dict and retval(retval.keys()) == 0:
        raise Exception('Returned dict is empty, must contain atleast 1 metric to maximize or minimize.')

    return_file = hdfs_exec_logdir + '/.return'
    hdfs.dump(retval, return_file)

    # Get the metric from dict from dict or directly returned value
    if optimization_key and type(retval) is dict:
        metric = retval[optimization_key]
    elif type(retval) is dict and retval(retval.keys()) == 1:
        metric = retval[retval.keys()[0]]
    else:
        metric = retval

    try:
        metric = int(metric)
    except:
        raise ValueError('Metric to maximize or minimize is not a number')

    metric_file = hdfs_exec_logdir + '/.metric'

    hdfs.dump(metric, metric_file)

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
        hyperparameters.append({'metrics': [{'key':'metric', 'value': metric}], 'hyperparameters': hp_arr})

    return json.dumps({'results': hyperparameters})

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
        best_param_arr.append({'key': hp[0], 'value': hp[1]})

    return best_param_arr

def _find_spark():
    """

    Returns: SparkSession

    """
    return SparkSession.builder.getOrCreate()


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

def _publish_experiment(app_id, run_id, json_data, xattr):
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
                   app_id + "_" + str(run_id) + "?xattr=" + xattr

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

def _finalize_experiment(experiment_json, hyperparameter, metric, state, duration):
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

    #experiment_json['hyperparameter'] = hyperparameter

    return json.dumps(experiment_json)

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