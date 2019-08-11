"""
Gridsearch implementation
"""

import os
from hops import hdfs as hopshdfs
from hops import tensorboard
from hops import devices
from hops import util

import pydoop.hdfs
import threading
import six
import time

def _grid_launch(sc, map_fun, run_id, args_dict, direction='max', local_logdir=False, name="no-name"):
    """
    Run the wrapper function with each hyperparameter combination as specified by the dictionary

    Args:
        sc:
        map_fun:
        args_dict:
        direction:
        local_logdir:
        name:

    Returns:

    """
    app_id = str(sc.applicationId)
    num_executions = 1

    if direction != 'max' and direction != 'min':
        raise ValueError('Invalid direction ' + direction +  ', must be max or min')

    arg_lists = list(args_dict.values())
    currentLen = len(arg_lists[0])
    for i in range(len(arg_lists)):
        if currentLen != len(arg_lists[i]):
            raise ValueError('Length of each function argument list must be equal')
        num_executions = len(arg_lists[i])

    #Each TF task should be run on 1 executor
    nodeRDD = sc.parallelize(range(num_executions), num_executions)

    #Make SparkUI intuitive by grouping jobs
    sc.setJobGroup("Grid Search", "{} | Hyperparameter Optimization".format(name))

    #Force execution on executor, since GPU is located on executor
    job_start = time.time()
    nodeRDD.foreachPartition(_prepare_func(app_id, run_id, map_fun, args_dict, local_logdir))
    job_end = time.time()

    job_time_str = util._time_diff(job_start, job_end)

    arg_count = six.get_function_code(map_fun).co_argcount
    arg_names = six.get_function_code(map_fun).co_varnames
    hdfs_dir = util._get_logdir(app_id, run_id)

    max_val, max_hp, min_val, min_hp, avg = _get_best(args_dict, num_executions, arg_names, arg_count, hdfs_dir, run_id)

    param_combination = ""
    best_val = ""

    if direction == 'max':
        param_combination = max_hp
        best_val = str(max_val)
        results = '\n------ Grid Search results ------ direction(' + direction + ') \n' \
          'BEST combination ' + max_hp + ' -- metric ' + str(max_val) + '\n' \
          'WORST combination ' + min_hp + ' -- metric ' + str(min_val) + '\n' \
          'AVERAGE metric -- ' + str(avg) + '\n' \
          'Total job time ' + job_time_str + '\n'
        _write_result(hdfs_dir, results)
        print(results)
    elif direction == 'min':
        param_combination = min_hp
        best_val = str(min_val)
        results = '\n------ Grid Search results ------ direction(' + direction + ') \n' \
        'BEST combination ' + min_hp + ' -- metric ' + str(min_val) + '\n' \
        'WORST combination ' + max_hp + ' -- metric ' + str(max_val) + '\n' \
        'AVERAGE metric -- ' + str(avg) + '\n' \
        'Total job time ' + job_time_str + '\n'
        _write_result(hdfs_dir, results)
        print(results)


    print('Finished Experiment \n')

    return hdfs_dir, param_combination, best_val

def _write_result(runid_dir, string):
    """

    Args:
        runid_dir:
        string:

    Returns:

    """
    metric_file = runid_dir + '/summary'
    fs_handle = hopshdfs.get_fs()
    try:
        fd = fs_handle.open_file(metric_file, mode='w')
    except:
        fd = fs_handle.open_file(metric_file, flags='w')
    fd.write(string.encode())
    fd.flush()
    fd.close()

def _prepare_func(app_id, run_id, map_fun, args_dict, local_logdir):
    """

    Args:
        app_id:
        run_id:
        map_fun:
        args_dict:
        local_logdir:

    Returns:

    """

    def _wrapper_fun(iter):
        """

        Args:
            iter:

        Returns:

        """

        for i in iter:
            executor_num = i

        tb_hdfs_path = ''
        hdfs_exec_logdir = ''

        t = threading.Thread(target=devices._print_periodic_gpu_utilization)
        if devices.get_num_gpus() > 0:
            t.start()

        try:
            #Arguments
            if args_dict:
                argcount = six.get_function_code(map_fun).co_argcount
                names = six.get_function_code(map_fun).co_varnames

                args = []
                argIndex = 0
                param_string = ''
                while argcount > 0:
                    #Get args for executor and run function
                    param_name = names[argIndex]
                    param_val = args_dict[param_name][executor_num]
                    param_string += str(param_name) + '=' + str(param_val) + '.'
                    args.append(param_val)
                    argcount -= 1
                    argIndex += 1
                param_string = param_string[:-1]
                hdfs_exec_logdir, hdfs_appid_logdir = util._create_experiment_subdirectories(app_id, run_id, param_string, 'grid_search')
                tb_hdfs_path, tb_pid = tensorboard._register(hdfs_exec_logdir, hdfs_appid_logdir, executor_num, local_logdir=local_logdir)

                gpu_str = '\nChecking for GPUs in the environment' + devices._get_gpu_info()
                print(gpu_str)
                print('-------------------------------------------------------')
                print('Started running task ' + param_string + '\n')
                task_start = time.time()
                retval = map_fun(*args)
                task_end = time.time()
                _handle_return(retval, hdfs_exec_logdir)
                time_str = 'Finished task ' + param_string + ' - took ' + util._time_diff(task_start, task_end)
                print('\n' + time_str)
                print('Returning metric ' + str(retval))
                print('-------------------------------------------------------')
        except:
            raise
        finally:
            util.cleanup(tensorboard.local_logdir_bool, tensorboard.local_logdir_path, hdfs_exec_logdir, t, tb_hdfs_path)

    return _wrapper_fun


def _get_best(args_dict, num_combinations, arg_names, arg_count, hdfs_appid_dir, run_id):
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

    max_hp = ''
    max_val = ''

    min_hp = ''
    min_val = ''

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
            param_string += str(param_name) + '=' + str(param_val) + '.'
            num_args -= 1
            argIndex += 1

        param_string = param_string[:-1]

        path_to_metric = hdfs_appid_dir + '/' + param_string + '/metric'

        with pydoop.hdfs.open(path_to_metric, "r") as fi:
            metric = float(fi.read())
            fi.close()

            if first:
                max_hp = param_string
                max_val = metric
                min_hp = param_string
                min_val = metric
                first = False

            if metric > max_val:
                max_val = metric
                max_hp = param_string
            if metric <  min_val:
                min_val = metric
                min_hp = param_string


        results.append(metric)

    avg = sum(results)/float(len(results))

    return max_val, max_hp, min_val, min_hp, avg


def _handle_return(val, hdfs_exec_logdir):
    """

    Args:
        val:
        hdfs_exec_logdir:

    Returns:

    """
    try:
        test = int(val)
    except:
        raise ValueError('Your function needs to return a metric (number) which should be maximized or minimized')

    metric_file = hdfs_exec_logdir + '/metric'
    fs_handle = hopshdfs.get_fs()
    try:
        fd = fs_handle.open_file(metric_file, mode='w')
    except:
        fd = fs_handle.open_file(metric_file, flags='w')
    fd.write(str(float(val)).encode())
    fd.flush()
    fd.close()