"""

Utility functions to manage the lifecycle of TensorFlow Extended (TFX) and Beam.

"""
import subprocess
import json
import os
import socket
from random import randint
import time
import atexit

from hops import constants
from hops import util
from hops import hdfs as hopsfs
from hops import jobs
from hops.exceptions import RestAPIError

# Dict of running jobserver and their flink clusters
clusters = []
kill_runners = False
jobserver_host = ""
jobserver_port = -1


def start_beam_jobserver(flink_session_name,
                         artifacts_dir="Resources",
                         jobserver_jar=os.path.join(util.get_flink_conf_dir(),
                                                     "beam-runners-flink-1.8-job-server-2.15.0.jar"),
                         sdk_worker_parallelism=1):
    """
    Start the Java Beam job server that connects to the flink session cluster. User needs to provide the
    job name that started the Flink session and optionally the worker parallelism.

    Args:
      :flink_session_name: Job name that runs the Flink session.
      :sdk_worker_parallelism: Default parallelism for SDK worker processes. This option is only applied when the
      pipeline option sdkWorkerParallelism is set to 0.Default is 1, If 0, worker parallelism will be dynamically
      decided by runner.See also: sdkWorkerParallelism Pipeline Option (default: 1). For further documentation,
      please refer to Apache Beam docs.
    Returns:
        artifact_port, expansion_port, job_host, job_port, jobserver.pid
    """
    # Get Flink master URL (flink session cluster) from an ExecutionDTO
    method = constants.HTTP_CONFIG.HTTP_GET
    connection = util._get_http_connection(https=True)
    resource_url = constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_REST_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_PROJECT_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   hopsfs.project_id() + constants.DELIMITERS.SLASH_DELIMITER + \
                   "jobs" + constants.DELIMITERS.SLASH_DELIMITER + \
                   flink_session_name + constants.DELIMITERS.SLASH_DELIMITER + \
                   "executions" + \
                   "?limit=1&offset=0&sort_by=submissionTime:desc"
    response = util.send_request(connection, method, resource_url)
    resp_body = response.read()
    flink_master_url = json.loads(resp_body)['items'][0]['flinkMasterURL']
    artifact_port = randint(10000, 65000)
    expansion_port = randint(10000, 65000)
    job_port = randint(10000, 65000)
    job_host = socket.getfqdn()
    log_base_path = ""
    if 'LOG_DIRS' in os.environ:
        log_base_path += os.environ['LOG_DIRS'] + "/"

    beam_jobserver_log = log_base_path + "beamjobserver-" + hopsfs.project_name() + "-" + flink_session_name + \
                          "-" + str(job_port) + ".log"
    # copy jar to local
    with open(beam_jobserver_log, "wb") as out, open(beam_jobserver_log, "wb") as err:
        jobserver = subprocess.Popen(["java",
                                       "-jar", jobserver_jar,
                                       "--artifacts-dir=%s" % hopsfs.project_path() + artifacts_dir,
                                       "--flink-master-url=%s" % flink_master_url,
                                       "--artifact-port=%d" % artifact_port,
                                       "--expansion-port=%d" % expansion_port,
                                       "--job-host=%s" % job_host,
                                       "--job-port=%d" % job_port,
                                       "--sdk-worker-parallelism=%d" % sdk_worker_parallelism],
                                      stdout=out,
                                      stderr=err,
                                      preexec_fn=util._on_executor_exit('SIGTERM'))
    global clusters
    clusters.append(flink_session_name)
    global jobserver_host
    jobserver_host = job_host
    global jobserver_port
    jobserver_port = job_port
    return {"jobserver_log": beam_jobserver_log,
            "artifact_port": artifact_port,
            "expansion_port": expansion_port,
            "job_host": job_host,
            "job_port": job_port,
            "jobserver.pid": jobserver.pid}


@atexit.register
def exit_handler():
    global clusters
    global kill_runners
    if kill_runners:
        for cluster in clusters:
            stop_runner(cluster)


def get_sdk_worker():
    """
    Get the path to the portability framework SDK worker script.

    Returns:
        the path to sdk_worker.sh
    """
    return os.path.join(util.get_flink_conf_dir(), "sdk_worker.sh")


def create_runner(runner_name, jobmanager_heap_size=1024, num_of_taskmanagers=1, taskmanager_heap_size=4096,
                  num_task_slots=1):
    """
    Create a Beam runner. Creates the job with the job type that corresponds to the requested runner

    Args:
        runner_name: Name of the runner.
        jobmanager_heap_size: The memory(mb) of the Flink cluster JobManager
        num_of_taskmanagers: The number of TaskManagers of the Flink cluster.
        taskmanager_heap_size: The memory(mb) of the each TaskManager in the Flink cluster.
        num_task_slots: Number of slots of the Flink cluster.

    Returns:
        The runner spec.
    """

    # In the future we will support beamSparkJobConfiguration
    type = "beamFlinkJobConfiguration"
    job_config = {"type": type,
                  "amQueue": "default",
                  "jobmanager.heap.size": jobmanager_heap_size,
                  "amVCores": "1",
                  "numberOfTaskManagers": num_of_taskmanagers,
                  "taskmanager.heap.size": taskmanager_heap_size,
                  "taskmanager.numberOfTaskSlots": num_task_slots}
    return jobs.create_job(runner_name, job_config)


def start_runner(runner_name):
    """
    Start the runner. Submits an http request to the HOPSWORKS REST API to start the job

    Returns:
        The runner execution status.
    """
    # Auto-generate a runner name
    return jobs.start_job(runner_name)


def stop_runner(runner_name):
    """
    Stop the runner.

    Returns:
        The runner execution status.
    """
    # Auto-generate a runner name
    return jobs.stop_job(runner_name)


def start(runner_name="runner", jobmanager_heap_size=1024,
          num_of_taskmanagers=1, taskmanager_heap_size=4096, num_task_slots=1, kill_runner=False):
    """
    Creates and starts a Beam runner and then starts the beam job server.

    Args:
        runner_name: Name of the runner. If not specified, the default runner name "runner" will be used. If the
        runner already exists, it will be updated with the provided arguments. If it doesn't exist, it will be
        created.
        jobmanager_heap_size: The memory(mb) of the Flink cluster JobManager
        num_of_taskmanagers: The number of TaskManagers of the Flink cluster.
        taskmanager_heap_size: The memory(mb) of the each TaskManager in the Flink cluster.
        num_task_slots: Number of slots of the Flink cluster.
        kill_runner: Kill runner when Python terminates
    Returns:
        The artifact_port, expansion_port, job_host, job_port, jobserver.pid
    """
    global kill_runners
    kill_runners = kill_runner
    # If runner exists and is running, skip creating.
    execution_found = True
    try:
        execution = jobs.get_current_execution(runner_name)
        if 'items' in execution and execution['items'][0]['state'] != 'RUNNING':
            execution_found = False
    except RestAPIError as err:
        if "HTTP code: 404" in err.args[0]:
            execution_found = False

    if not execution_found:
        job = create_runner(runner_name, jobmanager_heap_size, num_of_taskmanagers, taskmanager_heap_size, num_task_slots)
        start_runner(runner_name)
        # Wait 90 seconds until runner is in status "RUNNING", then start the jobserver
        wait = 90
        wait_count = 0
        while wait_count < wait and jobs.get_current_execution(job['name'])['items'][0]['state'] != "RUNNING":
            time.sleep(5)
            wait_count += 5

    return start_beam_jobserver(runner_name)


def get_portable_runner_config(sdk_worker_parallelism=1, worker_threads=100, pre_optimize="all",
                               execution_mode_for_batch="BATCH_FORCED"):
    """
    Instantiate a list of pipeline configuration options for the PortableRunner.

    Args:
        sdk_worker_parallelism: sdk_worker_parallelism
        worker_threads: worker_threads
        pre_optimize: pre_optimize
        execution_mode_for_batch: execution_mode_for_batch

    Returns:
        a list of pipeline configuration options for the PortableRunner.
    """
    return ['--runner=PortableRunner',
            '--hdfs_host=' + str(hopsfs.get_webhdfs_host()),
            '--hdfs_port=' + str(hopsfs.get_webhdfs_port()),
            '--hdfs_user=' + hopsfs.project_user(),
            '--job_endpoint=' + jobserver_host + ":" + str(jobserver_port),
            '--environment_type=PROCESS',
            '--environment_config=' + '{"command":"' + get_sdk_worker() + '"}',
            '--sdk_worker_parallelism=' + str(sdk_worker_parallelism),
            '--experiments=worker_threads' + str(worker_threads),
            '--experiments=pre_optimize=' + pre_optimize,
            '--execution_mode_for_batch=' + execution_mode_for_batch]
