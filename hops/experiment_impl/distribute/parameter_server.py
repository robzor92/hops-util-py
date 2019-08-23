"""
Utility functions to retrieve information about available services and setting up security for the Hops platform.

These utils facilitates development by hiding complexity for programs interacting with Hops services.
"""

import os
from hops import devices, tensorboard
from hops.experiment_impl.util import experiment_utils
from hops import util

import pydoop.hdfs
import threading
import time
import socket
import json

from . import parameter_server_reservation

def _run(sc, map_fun, run_id, local_logdir=False, name="no-name", evaluator=False):
    """

    Args:
        sc:
        map_fun:
        local_logdir:
        name:

    Returns:

    """
    app_id = str(sc.applicationId)

    num_executions = util.num_executors()

    #Each TF task should be run on 1 executor
    nodeRDD = sc.parallelize(range(num_executions), num_executions)

    #Make SparkUI intuitive by grouping jobs
    sc.setJobGroup("ParameterServerStrategy", "{} | Distributed Training".format(name))

    server = parameter_server_reservation.Server(num_executions)

    server_addr = server.start()

    num_ps = util.num_param_servers()

    #Force execution on executor, since GPU is located on executor
    nodeRDD.foreachPartition(_prepare_func(app_id, run_id, map_fun, local_logdir, server_addr, num_ps, evaluator))

    logdir = experiment_utils._get_logdir(app_id, run_id)

    print('Finished Experiment \n')

    path_to_return = logdir + '/.return'
    if pydoop.hdfs.path.exists(path_to_return):
        with pydoop.hdfs.open(path_to_return, "r") as fi:
            contents = fi.read()
            fi.close()
            return logdir, json.loads(contents)

    return logdir, None

def _prepare_func(app_id, run_id, map_fun, local_logdir, server_addr, num_ps, evaluator):
    """

    Args:
        app_id:
        run_id:
        map_fun:
        local_logdir:
        server_addr:
        num_ps:

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

        t = threading.Thread(target=devices._print_periodic_gpu_utilization)
        if devices.get_num_gpus() > 0:
            t.start()

        role = None

        client = parameter_server_reservation.Client(server_addr)

        try:
            host = experiment_utils._get_ip_address()

            tmp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tmp_socket.bind(('', 0))
            port = tmp_socket.getsockname()[1]
            host_port = host + ":" + str(port)

            exec_spec = {}
            if executor_num < num_ps:
                exec_spec["task_type"] = "ps"
            else:
                exec_spec["task_type"] = "worker"
            exec_spec["host_port"] = host_port
            exec_spec["gpus_present"] = devices.get_num_gpus() > 0

            client.register(exec_spec)

            cluster = client.await_reservations()

            tmp_socket.close()

            role, index = experiment_utils._find_task_and_index(host_port, cluster)

            cluster_spec = {}
            cluster_spec["cluster"] = cluster
            cluster_spec["task"] = {"type": role, "index": index}

            evaluator_node = None
            if evaluator:
                last_worker_index = len(cluster_spec["cluster"]["worker"])-1
                evaluator_node = cluster_spec["cluster"]["worker"][last_worker_index]
                cluster_spec["cluster"]["evaluator"] = [evaluator_node]
                del cluster_spec["cluster"]["worker"][last_worker_index]
                if evaluator_node == host_port:
                    role = "evaluator"
                    cluster_spec["task"] = {"type": "evaluator", "index": 0}

            print('TF_CONFIG: {} '.format(cluster_spec))
            os.environ["TF_CONFIG"] = json.dumps(cluster_spec)

            if role == "chief":
                logdir = experiment_utils._get_logdir(app_id, run_id)
                tb_hdfs_path, tb_pid = tensorboard._register(logdir, logdir, executor_num, local_logdir=local_logdir)
            elif role == "evaluator":
                logdir = experiment_utils._get_logdir(app_id, run_id)
                tensorboard.events_logdir = logdir
                
            gpu_str = '\nChecking for GPUs in the environment' + devices._get_gpu_info()
            print(gpu_str)
            print('-------------------------------------------------------')
            print('Started running task \n')
            task_start = time.time()

            retval=None
            if role == "ps":
                ps_thread = threading.Thread(target=lambda: map_fun())
                ps_thread.start()
                client.await_all_workers_finished()
            else:
                retval = map_fun()

            if retval and is_chief:
                experiment_utils._handle_return_simple(retval, experiment_utils._get_logdir(app_id, run_id))

            task_end = time.time()
            time_str = 'Finished task - took ' + experiment_utils._time_diff(task_start, task_end)
            print('\n' + time_str)
            print('-------------------------------------------------------')
        except:
            raise
        finally:
            if role != "ps":
                client.register_worker_finished()
            client.close()
            experiment_utils._cleanup(tensorboard.local_logdir_bool, tensorboard.local_logdir_path, logdir, t, tb_hdfs_path)
    return _wrapper_fun