"""
Simple experiment implementation
"""

from hops.experiment_impl.util import experiment_utils
from hops import devices, tensorboard, hdfs

import pydoop.hdfs
import threading
import six
import time
import json


def _run(sc, map_fun, run_id, args_dict=None, local_logdir=False, name="no-name"):
    """

    Args:
        sc:
        map_fun:
        args_dict:
        local_logdir:
        name:

    Returns:

    """

    app_id = str(sc.applicationId)


    if args_dict == None:
        num_executions = 1
    else:
        arg_lists = list(args_dict.values())
        currentLen = len(arg_lists[0])
        for i in range(len(arg_lists)):
            if currentLen != len(arg_lists[i]):
                raise ValueError('Length of each function argument list must be equal')
            num_executions = len(arg_lists[i])

    sc.setJobGroup("Launcher", "{} | Running experiment".format(name))
    #Each TF task should be run on 1 executor
    nodeRDD = sc.parallelize(range(num_executions), num_executions)

    #Force execution on executor, since GPU is located on executor
    nodeRDD.foreachPartition(_prepare_func(app_id, run_id, map_fun, args_dict, local_logdir))

    print('Finished Experiment \n')

    if args_dict == None:
        path_to_return = experiment_utils._get_logdir(app_id, run_id) + '/.return'
        if hdfs.exists(path_to_return):
            contents = json.loads(hdfs.load(path_to_return))
            return experiment_utils._get_logdir(app_id, run_id), None, contents
    elif num_executions == 1 and not args_dict == None:
        arg_count = six.get_function_code(map_fun).co_argcount
        arg_names = six.get_function_code(map_fun).co_varnames
        argIndex = 0
        param_string = ''
        while arg_count > 0:
            param_name = arg_names[argIndex]
            param_val = args_dict[param_name][0]
            param_string += str(param_name) + '=' + str(param_val) + '&'
            arg_count -= 1
            argIndex += 1
        param_string = param_string[:-1]
        path_to_metric = experiment_utils._get_logdir(app_id, run_id) + '/' + param_string + '/.return'
        if hdfs.exists(path_to_return):
            contents = json.loads(hdfs.load(path_to_return))
            return experiment_utils._get_logdir(app_id, run_id), param_string, contents
        else:
            return experiment_utils._get_logdir(app_id, run_id), param_string, None


    return experiment_utils._get_logdir(app_id, run_id), None, None

#Helper to put Spark required parameter iter in function signature
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

        experiment_utils._set_ml_id(app_id, run_id)

        tb_hdfs_path = ''

        hdfs_exec_logdir = experiment_utils._get_logdir(app_id, run_id)

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
                    param_string += str(param_name) + '=' + str(param_val) + '&'
                    args.append(param_val)
                    argcount -= 1
                    argIndex += 1
                param_string = param_string[:-1]
                tb_hdfs_path, tb_pid = tensorboard._register(hdfs_exec_logdir, hdfs_exec_logdir, executor_num, local_logdir=local_logdir)

                gpu_str = '\nChecking for GPUs in the environment' + devices._get_gpu_info()
                print(gpu_str)
                print('-------------------------------------------------------')
                print('Started running task ' + param_string + '\n')
                task_start = time.time()
                retval = map_fun(*args)
                task_end = time.time()
                if retval:
                    experiment_utils._handle_return_simple(retval, hdfs_exec_logdir)
                time_str = 'Finished task ' + param_string + ' - took ' + experiment_utils._time_diff(task_start, task_end)
                print('\n' + time_str)
                print('-------------------------------------------------------')
            else:
                tb_hdfs_path, tb_pid = tensorboard._register(hdfs_exec_logdir, hdfs_exec_logdir, executor_num, local_logdir=local_logdir)
                gpu_str = '\nChecking for GPUs in the environment' + devices._get_gpu_info()
                print(gpu_str)
                print('-------------------------------------------------------')
                print('Started running task\n')
                task_start = time.time()
                retval = map_fun()
                task_end = time.time()
                if retval:
                    experiment_utils._handle_return_simple(retval, hdfs_exec_logdir)
                time_str = 'Finished task - took ' + experiment_utils._time_diff(task_start, task_end)
                print('\n' + time_str)
                print('-------------------------------------------------------')
        except:
            raise
        finally:
            experiment_utils._cleanup(tensorboard.local_logdir_bool, tensorboard.local_logdir_path, hdfs_exec_logdir, t, tb_hdfs_path)

    return _wrapper_fun