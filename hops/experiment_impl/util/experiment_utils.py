"""
Utility functions for experiments
"""

import os
import signal
from ctypes import cdll
import itertools
import socket
import json
from hops import constants
from hops import devices
from hops import util
from hops import hdfs
import urllib3
urllib3.disable_warnings(urllib3.exceptions.SecurityWarning)

import pydoop.hdfs

def _handle_return(retval, hdfs_exec_logdir, optimization_key):
    """

    Args:
        val:
        hdfs_exec_logdir:

    Returns:

    """

    _upload_file_output(retval, hdfs_exec_logdir)

    # Validation
    if not optimization_key and type(retval) is dict and len(retval.keys()) > 1:
        raise Exception('Missing optimization_key argument, when returning multiple values in a dict the optimization_key argument must be set.')
    elif type(retval) is dict and optimization_key not in retval and len(retval.keys()) > 1:
        raise Exception('Specified optimization key {} not in returned dict'.format(optimization_key))
    elif type(retval) is dict and len(retval.keys()) == 0:
        raise Exception('Returned dict is empty, must contain atleast 1 metric to maximize or minimize.')

    # Validate that optimization_key is a number
    if type(retval) is dict and len(retval.keys()) > 1:
        opt_val = retval[optimization_key]
        opt_val = _validate_optimization_value(opt_val)
        retval[optimization_key] = opt_val
    elif type(retval) is dict and len(retval.keys()) == 1:
        opt_val = retval[list(retval.keys())[0]]
        opt_val = _validate_optimization_value(opt_val)
        retval[list(retval.keys())[0]] = opt_val
    else:
        opt_val = _validate_optimization_value(retval)
        retval = {'metric': opt_val}

    # Make sure all digits are strings
    for key in retval.keys():
        retval[key] = _cast_number_to_string(retval[key])


    return_file = hdfs_exec_logdir + '/.return'
    hdfs.dump(json.dumps(retval), return_file)

    metric_file = hdfs_exec_logdir + '/.metric'
    hdfs.dump(str(opt_val), metric_file)

def _validate_optimization_value(opt_val):
        try:
            int(opt_val)
            return opt_val
        except:
            pass
        try:
            float(opt_val)
            return opt_val
        except:
            pass
        raise ValueError('Metric to maximize or minimize is not a number: {}'.format(opt_val))

def _cast_number_to_string(val):
    is_number=False
    try:
        int(val)
        is_number=True
    except:
        pass
    try:
        float(val)
        is_number=True
    except:
        pass

    if is_number:
        return str(val)
    else:
        return val

def _upload_file_output(retval, hdfs_exec_logdir):
    if type(retval) is dict:
        for metric_key in retval.keys():
            value = str(retval[metric_key])
            if '/' in value or os.path.exists(os.getcwd() + '/' + value):
                if os.path.exists(value): # absolute path
                    if hdfs.exists(hdfs_exec_logdir + '/' + value.split('/')[-1]):
                        hdfs.delete(hdfs_exec_logdir + '/' + value.split('/')[-1], recursive=False)
                    pydoop.hdfs.put(value, hdfs_exec_logdir)
                    os.remove(value)
                    hdfs_exec_logdir = hdfs.abs_path(hdfs_exec_logdir)
                    retval[metric_key] = hdfs_exec_logdir[len(hdfs.abs_path(hdfs.project_path())):] + '/' +  value.split('/')[-1]
                elif os.path.exists(os.getcwd() + '/' + value): # relative path
                    output_file = os.getcwd() + '/' + value
                    if hdfs.exists(hdfs_exec_logdir + '/' + value):
                        hdfs.delete(hdfs_exec_logdir + '/' + value, recursive=False)
                    pydoop.hdfs.put(value, hdfs_exec_logdir)
                    os.remove(output_file)
                    hdfs_exec_logdir = hdfs.abs_path(hdfs_exec_logdir)
                    retval[metric_key] = hdfs_exec_logdir[len(hdfs.abs_path(hdfs.project_path())):] + '/' +  output_file.split('/')[-1]
                else:
                    raise Exception('Could not find file or directory or path ' + str(value))


def _handle_return_simple(retval, hdfs_exec_logdir):
    """

    Args:
        val:
        hdfs_exec_logdir:

    Returns:

    """

    if not retval:
        return

    _upload_file_output(retval, hdfs_exec_logdir)

    # Validation
    if type(retval) is not dict:
        try:
            metric = int(retval)
            retval = {'metric': retval}
        except:
            raise ValueError('Metric to maximize or minimize is not a number: {}'.format(retval))

    return_file = hdfs_exec_logdir + '/.return'
    hdfs.dump(json.dumps(retval), return_file)

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

def _build_summary_json(logdir):

    hyperparameters = []
    return_files = []

    for experiment_dir in hdfs.ls(logdir):
        runs = hdfs.ls(experiment_dir, recursive=True)
        for run in runs:
            if run.endswith('.return'):
                return_files.append(run)

    for return_file in return_files:
        return_file_abs = hdfs.abs_path(return_file)
        hyperparameter_combination = os.path.split(os.path.dirname(return_file_abs))[1]

        hp_arr = _convert_param_to_arr(hyperparameter_combination)
        metric_arr = _convert_return_file_to_arr(return_file)
        hyperparameters.append({'metrics': metric_arr, 'hyperparameters': hp_arr})

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
        if pyhdfs_handle.exists(hdfs_exec_logdir):
            hdfs.delete(hdfs_exec_logdir, recursive=True)
    elif not param_string and not sub_type:
        if pyhdfs_handle.exists(hdfs_experiment_dir):
            hdfs.delete(hdfs_experiment_dir, recursive=True)
        hdfs_exec_logdir = hdfs_experiment_dir + '/'
    else:
        hdfs_exec_logdir = hdfs_experiment_dir + '/' + str(param_string)
        if pyhdfs_handle.exists(hdfs_exec_logdir):
            hdfs.delete(hdfs_exec_logdir, recursive=True)

    # Need to remove directory if it exists (might be a task retry)

    # create the new directory
    pyhdfs_handle.create_directory(hdfs_exec_logdir)

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
    if '=' in best_param:
        best_param = best_param.split('&')
        for hp in best_param:
            hp = hp.split('=')
            best_param_arr.append({'key': hp[0], 'value': hp[1]})

    return best_param_arr

def _convert_return_file_to_arr(return_file_path):
    return_file_contents = hdfs.load(return_file_path)

    # Could be a number
    try:
        metric = int(return_file_contents)
        return [{'key':'metric', 'value': metric}]
    except:
        pass

    metrics_return_file = json.loads(return_file_contents)
    metrics_arr = []
    for metric_key in metrics_return_file:
        metrics_arr.append([{'key':metric_key, 'value': metrics_return_file[metric_key]}])

    return metrics_arr

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

def _attach_experiment_xattr(app_id, run_id, json_data, xattr):
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

    resp = util.send_request_with_session('POST', resource_url, data=json_data, headers=headers)
    print(resource_url)
    print(resp)

def _attach_model_xattr(ml_id, model, xattr):
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
    resource_url = constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_REST_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_PROJECT_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   hdfs.project_id() + constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_EXPERIMENTS_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   ml_id + "?xattr=" + xattr + "&model=" + model

    resp = util.send_request_with_session('POST', resource_url)
    print(resource_url)
    print(resp)

def _populate_experiment(model_name, function, type, hp, versioned_resources, description, app_id, direction, optimization_key):
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
    return json.dumps({'name': model_name, 'description': description, 'state': 'RUNNING', 'function': function, 'experimentType': type,
                       'appId': app_id, 'direction': direction, 'optimizationKey': optimization_key})

def _finalize_experiment_json(experiment_json, metric, state, duration):
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

def _get_best(args_dict, num_combinations, arg_names, arg_count, hdfs_appid_dir, optimization_key):
    """

    Args:
        args_dict:
        num_combinations:
        arg_names:
        arg_count:
        hdfs_appid_dir:
        run_id:

    Returns:

    """

    if not optimization_key:
        optimization_key = 'metric'

    max_hp = ''
    max_val = ''

    min_hp = ''
    min_val = ''

    min_return_dict = {}
    max_return_dict = {}

    results = []

    first = True

    for i in range(num_combinations):

        argIndex = 0
        param_string = ''

        num_args = arg_count

        while num_args > 0:
            #Get args for executor and run function
            param_name = arg_names[argIndex]
            param_val = args_dict[param_name][i]
            param_string += str(param_name) + '=' + str(param_val) + '&'
            num_args -= 1
            argIndex += 1

        param_string = param_string[:-1]

        path_to_return = hdfs_appid_dir + '/' + param_string + '/.return'

        assert hdfs.exists(path_to_return), 'Could not find .return file on path: {}'.format(path_to_return)

        with hdfs.open_file(path_to_return, flags="r") as fi:
            return_dict = json.loads(fi.read())
            fi.close()

            # handle case when dict with 1 key is returned
            if optimization_key == 'metric' and len(return_dict.keys()) == 1:
                optimization_key = list(return_dict.keys())[0]

            metric = float(return_dict[optimization_key])

            if first:
                max_hp = param_string
                max_val = metric
                max_return_dict = return_dict
                min_hp = param_string
                min_val = metric
                min_return_dict = return_dict
                first = False

            if metric > max_val:
                max_val = metric
                max_hp = param_string
                max_return_dict = return_dict
            if metric <  min_val:
                min_val = metric
                min_hp = param_string
                min_return_dict = return_dict

        results.append(metric)

    avg = sum(results)/float(len(results))

    return max_val, max_hp, min_val, min_hp, avg, max_return_dict, min_return_dict

def _setup_experiment(versioned_resources, logdir, app_id, run_id):
    versioned_path = _version_resources(versioned_resources, logdir)
    hdfs.mkdir(_get_logdir(app_id, run_id))
    return versioned_path

def _finalize_experiment(experiment_json, hp, metric, app_id, run_id, state, duration, logdir):

    outputs = _build_summary_json(logdir)

    if outputs:
        hdfs.dump(outputs, logdir + '/.summary')

    experiment_json = _finalize_experiment_json(experiment_json, metric, state, duration)

    _attach_experiment_xattr(app_id, run_id, experiment_json, 'REPLACE')

def _find_task_and_index(host_port, cluster_spec):
    """

    Args:
        host_port:
        cluster_spec:

    Returns:

    """
    index = 0
    for entry in cluster_spec["worker"]:
        if entry == host_port:
            return "worker", index
        index = index + 1

    index = 0
    for entry in cluster_spec["ps"]:
        if entry == host_port:
            return "ps", index
        index = index + 1


    if cluster_spec["chief"][0] == host_port:
        return "chief", 0

def _find_index(host_port, cluster_spec):
    """

    Args:
        host_port:
        cluster_spec:

    Returns:

    """
    index = 0
    for entry in cluster_spec["cluster"]["worker"]:
        if entry == host_port:
            return index
        else:
            index = index + 1
    return -1

def _set_ml_id(app_id, run_id):
    os.environ['ML_ID'] = str(app_id) + '_' + str(run_id)