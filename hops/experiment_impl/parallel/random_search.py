"""
Random Search implementation
"""

from hops.experiment_impl.util import experiment_utils
from hops.experiment_impl.tensorboard import tensorboard
from hops import devices

import threading
import six
import time
import random

def _run(sc, map_fun, run_id, args_dict, samples, direction='max', local_logdir=False, name="no-name", optimization_key=None):
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

    arg_lists = list(args_dict.values())
    for i in range(len(arg_lists)):
       if len(arg_lists[i]) != 2:
           raise ValueError('Boundary list must contain exactly two elements, [lower_bound, upper_bound] for each hyperparameter')

    hp_names = args_dict.keys()

    random_dict = {}
    for hp in hp_names:
        lower_bound = args_dict[hp][0]
        upper_bound = args_dict[hp][1]

        assert lower_bound < upper_bound, "lower bound: " + str(lower_bound) + " must be less than upper bound: " + str(upper_bound)

        random_values = []

        if type(lower_bound) == int and type(upper_bound) == int:
            for i in range(samples):
                random_values.append(random.randint(lower_bound, upper_bound))
        elif type(lower_bound) == float and type(upper_bound) == float:
            for i in range(samples):
                random_values.append(random.uniform(lower_bound, upper_bound))
        else:
            raise ValueError('Only float and int is currently supported')

        random_dict[hp] = random_values

    random_dict, new_samples = _remove_duplicates(random_dict, samples)

    sc.setJobGroup("Random Search", "{} | Hyperparameter Optimization".format(name))
    #Each TF task should be run on 1 executor
    nodeRDD = sc.parallelize(range(new_samples), new_samples)

    nodeRDD.foreachPartition(_prepare_func(app_id, run_id, map_fun, random_dict, local_logdir, optimization_key))

    arg_count = six.get_function_code(map_fun).co_argcount
    arg_names = six.get_function_code(map_fun).co_varnames
    exp_dir = experiment_utils._get_logdir(app_id, run_id)

    max_val, max_hp, min_val, min_hp, avg = experiment_utils._get_best(random_dict, new_samples, arg_names, arg_count, exp_dir)

    param_combination = ""
    best_val = ""

    if direction == 'max':
        param_combination = max_hp
        best_val = str(max_val)
    elif direction == 'min':
        param_combination = min_hp
        best_val = str(min_val)

    print('Finished Experiment \n')

    return exp_dir, param_combination, best_val

def _remove_duplicates(random_dict, samples):
    hp_names = random_dict.keys()
    concatenated_hp_combs_arr = []
    for index in range(samples):
        separated_hp_comb = ""
        for hp in hp_names:
            separated_hp_comb = separated_hp_comb + str(random_dict[hp][index]) + "&"
        concatenated_hp_combs_arr.append(separated_hp_comb)

    entry_index = 0
    indices_to_skip = []
    for entry in concatenated_hp_combs_arr:
        inner_index = 0
        for possible_dup_entry in concatenated_hp_combs_arr:
            if entry == possible_dup_entry and inner_index > entry_index:
                indices_to_skip.append(inner_index)
            inner_index = inner_index + 1
        entry_index = entry_index + 1
    indices_to_skip = list(set(indices_to_skip))

    for hp in hp_names:
        index = 0
        pruned_duplicates_arr = []
        for random_value in random_dict[hp]:
            if index not in indices_to_skip:
                pruned_duplicates_arr.append(random_value)
            index = index + 1
        random_dict[hp] = pruned_duplicates_arr

    return random_dict, samples - len(indices_to_skip)

#Helper to put Spark required parameter iter in function signature
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
                hdfs_exec_logdir, hdfs_appid_logdir = experiment_utils._create_experiment_subdirectories(app_id, run_id, param_string, 'random_search')
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