"""
Utility functions to retrieve information about available services and setting up security for the Hops platform.

These utils facilitates development by hiding complexity for programs interacting with Hops services.
"""

import os
from hops import hdfs as hopshdfs
from hops import tensorboard
from hops import devices
from hops import util

import pydoop.hdfs
import threading
import datetime
import socket
import json

from . import allreduce_reservation

def _launch(sc, map_fun, run_id, local_logdir=False, name="no-name", evaluator=False):
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
    sc.setJobGroup("CollectiveAllReduceStrategy", "{} | Distributed Training".format(name))

    server = allreduce_reservation.Server(num_executions)
    server_addr = server.start()

    #Force execution on executor, since GPU is located on executor
    nodeRDD.foreachPartition(_prepare_func(app_id, run_id, map_fun, local_logdir, server_addr, evaluator))

    logdir = _get_logdir(app_id, run_id)

    path_to_metric = logdir + '/metric'
    if pydoop.hdfs.path.exists(path_to_metric):
        with pydoop.hdfs.open(path_to_metric, "r") as fi:
            metric = float(fi.read())
            fi.close()
            return metric, logdir

    print('Finished Experiment \n')

    return None, logdir

def _get_logdir(app_id, run_id):
    """

    Args:
        app_id:

    Returns:

    """
    return util._get_experiments_dir() + '/' + app_id + '_' + str(run_id)

def _prepare_func(app_id, run_id, map_fun, local_logdir, server_addr, evaluator):
    """

    Args:
        app_id:
        run_id:
        map_fun:
        local_logdir:
        server_addr:

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

        t = threading.Thread(target=devices._print_periodic_gpu_utilization)
        if devices.get_num_gpus() > 0:
            t.start()

        try:
            host = util._get_ip_address()

            tmp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tmp_socket.bind(('', 0))
            port = tmp_socket.getsockname()[1]

            client = allreduce_reservation.Client(server_addr)
            host_port = host + ":" + str(port)

            client.register({"worker": host_port, "index": executor_num})
            cluster = client.await_reservations()
            tmp_socket.close()
            client.close()

            task_index = _find_index(host_port, cluster)

            if task_index == -1:
                cluster["task"] = {"type": "chief", "index": 0}
            else:
                cluster["task"] = {"type": "worker", "index": task_index}

            if evaluator:
                evaluator_node = cluster["cluster"]["worker"][0]
                cluster["cluster"]["evaluator"] = [evaluator_node]
                del cluster["cluster"]["worker"][0]
                if evaluator_node == host_port:
                    cluster["task"] = {"type": "evaluator", "index": 0}

            print('TF_CONFIG: {} '.format(cluster))

            if util.num_executors() > 1:
                os.environ["TF_CONFIG"] = json.dumps(cluster)

            is_chief = (task_index == -1 or util.num_executors() == 1) and not evaluator_node == host_port
            is_evaluator = evaluator_node == host_port

            if is_chief:
                logdir = _get_logdir(app_id, run_id)
                tb_hdfs_path, tb_pid = tensorboard._register(logdir, logdir, executor_num, local_logdir=local_logdir)
            elif is_evaluator:
                logdir = _get_logdir(app_id, run_id)
                tensorboard.events_logdir = logdir
            gpu_str = '\nChecking for GPUs in the environment' + devices._get_gpu_info()

            print(gpu_str)
            print('-------------------------------------------------------')
            print('Started running task \n')
            task_start = datetime.datetime.now()

            retval = map_fun()
            if is_chief:
                if retval:
                    _handle_return(retval, logdir)
            task_end = datetime.datetime.now()
            time_str = 'Finished task - took ' + util._time_diff(task_start, task_end)
            print('\n' + time_str)
            print('-------------------------------------------------------')
        except:
            raise
        finally:
            if is_chief:
                if local_logdir:
                    local_tb = tensorboard.local_logdir_path
                    util._store_local_tensorboard(local_tb, logdir)

            if devices.get_num_gpus() > 0:
                t.do_run = False
                t.join(20)

            _cleanup(tb_hdfs_path)

    return _wrapper_fun

def _cleanup(tb_hdfs_path):
    """

    Args:
        tb_hdfs_path:

    Returns:

    """
    handle = hopshdfs.get()
    if not tb_hdfs_path == None and not tb_hdfs_path == '' and handle.exists(tb_hdfs_path):
        handle.delete(tb_hdfs_path)

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
        raise ValueError('Your function should return a metric (number).')

    metric_file = hdfs_exec_logdir + '/metric'
    fs_handle = hopshdfs.get_fs()
    try:
        fd = fs_handle.open_file(metric_file, mode='w')
    except:
        fd = fs_handle.open_file(metric_file, flags='w')
    fd.write(str(float(val)).encode())
    fd.flush()
    fd.close()

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