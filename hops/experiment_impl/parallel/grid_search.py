"""
Gridsearch implementation
"""

from hops import hdfs as hopshdfs, tensorboard
from hops import devices

from hops.experiment_impl.util import experiment_utils

import threading
import six
import time

def _run(sc, map_fun, run_id, args_dict, direction='max', local_logdir=False, name="no-name", optimization_key=None):
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
    nodeRDD.foreachPartition(_prepare_func(app_id, run_id, map_fun, args_dict, local_logdir, optimization_key))

    arg_count = six.get_function_code(map_fun).co_argcount
    arg_names = six.get_function_code(map_fun).co_varnames
    hdfs_dir = experiment_utils._get_logdir(app_id, run_id)

    max_val, max_hp, min_val, min_hp, avg = experiment_utils._get_best(args_dict, num_executions, arg_names, arg_count, hdfs_dir, optimization_key)

    param_combination = ""
    best_val = ""

    if direction == 'max':
        param_combination = max_hp
        best_val = str(max_val)
    elif direction == 'min':
        param_combination = min_hp
        best_val = str(min_val)


    print('Finished Experiment \n')

    return hdfs_dir + '/' + param_combination, param_combination, best_val

def _prepare_func(app_id, run_id, map_fun, args_dict, local_logdir, optimization_key):
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
                    param_string += str(param_name) + '=' + str(param_val) + '&'
                    args.append(param_val)
                    argcount -= 1
                    argIndex += 1
                param_string = param_string[:-1]
                hdfs_exec_logdir, hdfs_appid_logdir = experiment_utils._create_experiment_subdirectories(app_id, run_id, param_string, 'grid_search')
                tb_hdfs_path, tb_pid = tensorboard._register(hdfs_exec_logdir, hdfs_appid_logdir, executor_num, local_logdir=local_logdir)

                gpu_str = '\nChecking for GPUs in the environment' + devices._get_gpu_info()
                print(gpu_str)
                print('-------------------------------------------------------')
                print('Started running task ' + param_string + '\n')
                task_start = time.time()
                retval = map_fun(*args)
                task_end = time.time()
                experiment_utils._handle_return(retval, hdfs_exec_logdir, optimization_key)
                time_str = 'Finished task ' + param_string + ' - took ' + experiment_utils._time_diff(task_start, task_end)
                print('\n' + time_str)
                print('Returning metric ' + str(retval))
                print('-------------------------------------------------------')
        except:
            raise
        finally:
            experiment_utils._cleanup(tensorboard.local_logdir_bool, tensorboard.local_logdir_path, hdfs_exec_logdir, t, tb_hdfs_path)

    return _wrapper_fun