"""
Experiment module used for running Experiments, Parallel Experiments and Distributed Training on Hopsworks.

The programming model is that you wrap the code to run inside a wrapper function.
Inside that wrapper function provide all imports and parts that make up your experiment, see examples below.
Whenever a function to run an experiment is invoked it is also registered in the Experiments service along with the provided information.

*Three different types of experiments*
    - Run a single standalone Experiment using the *launch* function.
    - Run Parallel Experiments performing hyperparameter optimization using *grid_search* or *differential_evolution*.
    - Run single or multi-machine Distributed Training using *parameter_server* or *collective_all_reduce*.

"""

from hops import hdfs as hopshdfs

from hops.experiment_impl import differential_evolution as diff_evo_impl, grid_search as grid_search_impl, random_search as r_search_impl, launcher as launcher
from hops.experiment_impl.distribute import allreduce as allreduce_impl, parameter_server as ps_impl, mirrored as mirrored_impl

from hops import tensorboard

from hops import util

from time import time
import atexit
import json

run_id = 1
app_id = None
experiment_json = None
running = False
driver_tensorboard_hdfs_path = None

def launch(map_fun, args_dict=None, name='no-name', local_logdir=False, versioned_resources=None, description=None):
    """

    *Experiment* or *Parallel Experiment*

    Run an Experiment contained in *map_fun* one time with no arguments or multiple times with different arguments if
    *args_dict* is specified.

    Example usage:

    >>> from hops import experiment
    >>> def train_nn():
    >>>    import tensorflow
    >>>    from hops import tensorboard
    >>>    logdir = tensorboard.logdir()
    >>>    # code for preprocessing, training and exporting model
    >>>    # optionally return a value for the experiment which is registered in Experiments service
    >>> experiment.launch(train_nn)

    Args:
        :map_fun: The function to run
        :args_dict: If specified will run the same function multiple times with different arguments, {'a':[1,2], 'b':[5,3]}
         would run the function two times with arguments (1,5) and (2,3) provided that the function signature contains two arguments like *def func(a,b):*
        :name: name of the experiment
        :local_logdir: True if *tensorboard.logdir()* should be in the local filesystem, otherwise it is in HDFS
        :versioned_resources: A list of HDFS paths of resources to version with this experiment
        :description: A longer description for the experiment

    Returns:
        HDFS path in your project where the experiment is stored

    """

    num_ps = util.num_param_servers()
    assert num_ps == 0, "number of parameter servers should be 0"

    global running
    if running:
        raise RuntimeError("An experiment is currently running. Please call experiment.end() to stop it.")

    try:
        global app_id
        global experiment_json
        global run_id
        running = True

        sc = util._find_spark().sparkContext
        app_id = str(sc.applicationId)

        versioned_path = _setup_experiment(versioned_resources, launcher._get_logdir(app_id, run_id), app_id, run_id)

        experiment_json = None
        if args_dict:
            experiment_json = util._populate_experiment(sc, name, 'experiment', 'launcher', json.dumps(args_dict), versioned_path, description)
        else:
            experiment_json = util._populate_experiment(sc, name, 'experiment', 'launcher', None, versioned_path, description)

        util._publish_experiment(app_id, run_id, experiment_json, 'CREATE')

        retval, tensorboard_logdir, hp = launcher._launch(sc, map_fun, run_id, args_dict, local_logdir)

        if retval:
            experiment_json = util._finalize_experiment(experiment_json, hp, retval)
            util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')
            return tensorboard_logdir
    except:
        _exception_handler()
        raise
    finally:
        #cleanup spark jobs
        run_id +=1
        running = False
        sc.setJobGroup("", "")
    return tensorboard_logdir


def random_search(map_fun, boundary_dict, direction='max', samples=10, name='no-name', local_logdir=False, versioned_resources=None, description=None):
    """

    *Parallel Experiment*

    Run an Experiment contained in *map_fun* for configured number of random samples controlled by the *samples* parameter. Each hyperparameter is contained in *boundary_dict* with the key
    corresponding to the name of the hyperparameter and a list containing two elements defining the lower and upper bound.
    The experiment must return a metric corresponding to how 'good' the given hyperparameter combination is.

    Example usage:

    >>> from hops import experiment
    >>> boundary_dict = {'learning_rate': [0.1, 0.3], 'layers': [2, 9], 'dropout': [0.1,0.9]}
    >>> def train_nn(learning_rate, layers, dropout):
    >>>    import tensorflow
    >>>    # code for preprocessing, training and exporting model
    >>>    # mandatory return a value for the experiment which is registered in Experiments service
    >>>    return network.evaluate(learning_rate, layers, dropout)
    >>> experiment.random_search(train_nn, boundary_dict, samples=14, direction='max')

    Args:
        :map_fun: The function to run
        :boundary_dict: dict containing hyperparameter name and corresponding boundaries, each experiment randomize a value in the boundary range.
        :direction: If set to 'max' the highest value returned will correspond to the best solution, if set to 'min' the opposite is true
        :samples: the number of random samples to evaluate for each hyperparameter given the boundaries
        :name: name of the experiment
        :local_logdir: True if *tensorboard.logdir()* should be in the local filesystem, otherwise it is in HDFS
        :versioned_resources: A list of HDFS paths of resources to version with this experiment
        :description: A longer description for the experiment

    Returns:
        HDFS path in your project where the experiment is stored

    """

    num_ps = util.num_param_servers()
    assert num_ps == 0, "number of parameter servers should be 0"

    global running
    if running:
        raise RuntimeError("An experiment is currently running. Please call experiment.end() to stop it.")

    try:
        global app_id
        global experiment_json
        global run_id
        running = True

        sc = util._find_spark().sparkContext
        app_id = str(sc.applicationId)

        r_search_impl.run_id = run_id

        versioned_path = _setup_experiment(versioned_resources, r_search_impl._get_logdir(app_id, run_id), app_id, run_id)

        experiment_json = None

        experiment_json = util._populate_experiment(sc, name, 'experiment', 'random_search', json.dumps(boundary_dict), versioned_path, description)

        util._version_resources(versioned_resources, r_search_impl._get_logdir(app_id, run_id))

        util._publish_experiment(app_id, run_id, experiment_json, 'CREATE')

        tensorboard_logdir, param, metric = r_search_impl._launch(sc, map_fun, run_id, boundary_dict, samples, direction=direction, local_logdir=local_logdir)

        experiment_json = util._finalize_experiment(experiment_json, param, metric)

        util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')

        return tensorboard_logdir

    except:
        _exception_handler()
        raise
    finally:
        #cleanup spark jobs
        run_id +=1
        running = False
        sc.setJobGroup("", "")
    return tensorboard_logdir


def differential_evolution(objective_function, boundary_dict, direction = 'max', generations=10, population=10, mutation=0.5, crossover=0.7, cleanup_generations=False, name='no-name', local_logdir=False, versioned_resources=None, description=None):
    """
    *Parallel Experiment*

    Run differential evolution to explore a given search space for each hyperparameter and figure out the best hyperparameter combination.
    The function is treated as a blackbox that returns a metric for some given hyperparameter combination.
    The returned metric is used to evaluate how 'good' the hyperparameter combination was.

    Example usage:

    >>> from hops import experiment
    >>> boundary_dict = {'learning_rate':[0.01, 0.2], 'dropout': [0.1, 0.9]}
    >>> def train_nn(learning_rate, dropout):
    >>>    import tensorflow
    >>>    # code for preprocessing, training and exporting model
    >>>    # mandatory return a value for the experiment which is registered in Experiments service
    >>>    return network.evaluate(learning_rate, dropout)
    >>> experiment.differential_evolution(train_nn, boundary_dict, direction='max')

    Args:
        :objective_function: the function to run, must return a metric
        :boundary_dict: a dict where each key corresponds to an argument of *objective_function* and the correspond value should be a list of two elements. The first element being the lower bound for the parameter and the the second element the upper bound.
        :direction: 'max' to maximize the returned metric, 'min' to minize the returned metric
        :generations: number of generations
        :population: size of population
        :mutation: mutation rate to explore more different hyperparameters
        :crossover: how fast to adapt the population to the best in each generation
        :cleanup_generations: remove previous generations from HDFS, only keep the last 2
        :name: name of the experiment
        :local_logdir: True if *tensorboard.logdir()* should be in the local filesystem, otherwise it is in HDFS
        :versioned_resources: A list of HDFS paths of resources to version with this experiment
        :description: a longer description for the experiment

    Returns:
        HDFS path in your project where the experiment is stored, dict with best hyperparameters

    """

    num_ps = util.num_param_servers()
    assert num_ps == 0, "number of parameter servers should be 0"

    global running
    if running:
        raise RuntimeError("An experiment is currently running. Please call experiment.end() to stop it.")

    try:
        global app_id
        global experiment_json
        global run_id
        running = True
        spark = util._find_spark()
        sc = spark.sparkContext
        app_id = str(sc.applicationId)

        diff_evo_impl.run_id = run_id

        versioned_path = _setup_experiment(versioned_resources, diff_evo_impl._get_logdir(app_id, run_id), app_id, run_id)

        experiment_json = None
        experiment_json = util._populate_experiment(sc, name, 'experiment', 'differential_evolution', json.dumps(boundary_dict), versioned_path, description)

        util._publish_experiment(app_id, run_id, experiment_json, 'CREATE')

        tensorboard_logdir, best_param, best_metric = diff_evo_impl._search(spark, objective_function, run_id, boundary_dict, direction=direction, generations=generations, popsize=population, mutation=mutation, crossover=crossover, cleanup_generations=cleanup_generations, local_logdir=local_logdir, name=name)

        experiment_json = util._finalize_experiment(experiment_json, best_param, best_metric)

        util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')

        best_param_dict = util._convert_to_dict(best_param)

    except:
        _exception_handler()
        raise
    finally:
        #cleanup spark jobs
        run_id +=1
        running = False
        sc.setJobGroup("", "")

    return tensorboard_logdir, best_param_dict

def grid_search(map_fun, args_dict, direction='max', name='no-name', local_logdir=False, versioned_resources=None, description=None):
    """
    *Parallel Experiment*

    Run multiple experiments and test a grid of hyperparameters for a neural network to maximize e.g. a Neural Network's accuracy.

    The following example will run *train_nn* with 6 different hyperparameter combinations

    >>> from hops import experiment
    >>> grid_dict = {'learning_rate':[0.1, 0.3], 'dropout': [0.4, 0.6, 0.1]}
    >>> def train_nn(learning_rate, dropout):
    >>>    import tensorflow
    >>>    # code for preprocessing, training and exporting model
    >>>    # mandatory return a value for the experiment which is registered in Experiments service
    >>>    return network.evaluate(learning_rate, dropout)
    >>> experiment.grid_search(train_nn, grid_dict, direction='max')

    The following values will be injected in the function and run and evaluated.

        - (learning_rate=0.1, dropout=0.4)
        - (learning_rate=0.1, dropout=0.6)
        - (learning_rate=0.1, dropout=0.1)
        - (learning_rate=0.3, dropout=0.4)
        - (learning_rate=0.3, dropout=0.6)
        - (learning_rate=0.3, dropout=0.1)

    Args:
        :map_fun: the function to run, must return a metric
        :args_dict: a dict with a key for each argument with a corresponding value being a list containing the hyperparameters to test, internally all possible combinations will be generated and run as separate Experiments
        :direction: 'max' to maximize the returned metric, 'min' to minize the returned metric
        :name: name of the experiment
        :local_logdir: True if *tensorboard.logdir()* should be in the local filesystem, otherwise it is in HDFS
        :versioned_resources: A list of HDFS paths of resources to version with this experiment
        :description: a longer description for the experiment

    Returns:
        HDFS path in your project where the experiment is stored

    """

    num_ps = util.num_param_servers()
    assert num_ps == 0, "number of parameter servers should be 0"

    global running
    if running:
        raise RuntimeError("An experiment is currently running. Please call experiment.end() to stop it.")

    try:
        global app_id
        global experiment_json
        global run_id
        running = True

        sc = util._find_spark().sparkContext
        app_id = str(sc.applicationId)

        versioned_path = _setup_experiment(versioned_resources, grid_search_impl._get_logdir(app_id, run_id), app_id, run_id)

        experiment_json = util._populate_experiment(sc, name, 'experiment', 'grid_search', json.dumps(args_dict), versioned_path, description)

        util._publish_experiment(app_id, run_id, experiment_json, 'CREATE')

        grid_params = util.grid_params(args_dict)

        tensorboard_logdir, param, metric = grid_search_impl._grid_launch(sc, map_fun, run_id, grid_params, direction=direction, local_logdir=local_logdir, name=name)

        experiment_json = util._finalize_experiment(experiment_json, param, metric)

        util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')
    except:
        _exception_handler()
        raise
    finally:
        #cleanup spark jobs
        run_id +=1
        running = False
        sc.setJobGroup("", "")

    return tensorboard_logdir

def collective_all_reduce(map_fun, name='no-name', local_logdir=False, versioned_resources=None, description=None, evaluator=False):
    """
    *Distributed Training*

    Sets up the cluster to run CollectiveAllReduceStrategy.

    TF_CONFIG is exported in the background and does not need to be set by the user themselves.

    Example usage:

    >>> from hops import experiment
    >>> def distributed_training():
    >>>    import tensorflow
    >>>    from hops import tensorboard
    >>>    from hops import devices
    >>>    logdir = tensorboard.logdir()
    >>>    ...CollectiveAllReduceStrategy(num_gpus_per_worker=devices.get_num_gpus())...
    >>> experiment.collective_all_reduce(distributed_training, local_logdir=True)

    Args:
        :map_fun: the function containing code to run CollectiveAllReduceStrategy
        :name: the name of the experiment
        :local_logdir: True if *tensorboard.logdir()* should be in the local filesystem, otherwise it is in HDFS
        :versioned_resources: A list of HDFS paths of resources to version with this experiment
        :description: a longer description for the experiment

    Returns:
        HDFS path in your project where the experiment is stored

    """

    num_ps = util.num_param_servers()
    num_executors = util.num_executors()

    assert num_ps == 0, "number of parameter servers should be 0"
    assert num_executors > 1, "number of workers (executors) should be greater than 1"
    if evaluator:
        assert num_executors > 2, "number of workers must be atleast 3 if evaluator role is required"

    global running
    if running:
        raise RuntimeError("An experiment is currently running. Please call experiment.end() to stop it.")

    try:
        global app_id
        global experiment_json
        global run_id
        running = True

        sc = util._find_spark().sparkContext
        app_id = str(sc.applicationId)

        versioned_path = _setup_experiment(versioned_resources, allreduce_impl._get_logdir(app_id, run_id), app_id, run_id)

        experiment_json = util._populate_experiment(sc, name, 'experiment', 'collective_all_reduce', None, versioned_path, description)

        util._publish_experiment(app_id, run_id, experiment_json, 'CREATE')

        retval, logdir = allreduce_impl._launch(sc, map_fun, run_id, local_logdir=local_logdir, name=name, evaluator=evaluator)

        experiment_json = util._finalize_experiment(experiment_json, None, retval)

        util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')
    except:
        _exception_handler()
        raise
    finally:
        #cleanup spark jobs
        run_id +=1
        running = False
        sc.setJobGroup("", "")

    return logdir

def parameter_server(map_fun, name='no-name', local_logdir=False, versioned_resources=None, description=None, evaluator=False):
    """
    *Distributed Training*

    Sets up the cluster to run ParameterServerStrategy.

    TF_CONFIG is exported in the background and does not need to be set by the user themselves.

    Example usage:

    >>> from hops import experiment
    >>> def distributed_training():
    >>>    import tensorflow
    >>>    from hops import tensorboard
    >>>    from hops import devices
    >>>    logdir = tensorboard.logdir()
    >>>    ...ParameterServerStrategy(num_gpus_per_worker=devices.get_num_gpus())...
    >>> experiment.parameter_server(distributed_training, local_logdir=True)

    Args:
        :map_fun: contains the code where you are using ParameterServerStrategy.
        :name: name of the experiment
        :local_logdir: True if *tensorboard.logdir()* should be in the local filesystem, otherwise it is in HDFS
        :versioned_resources: A list of HDFS paths of resources to version with this experiment
        :description: a longer description for the experiment

    Returns:
        HDFS path in your project where the experiment is stored

    """
    num_ps = util.num_param_servers()
    num_executors = util.num_executors()

    assert num_ps > 0, "number of parameter servers should be greater than 0"
    assert num_ps < num_executors, "num_ps cannot be greater than num_executors (i.e. num_executors == num_ps + num_workers)"
    if evaluator:
        assert num_executors - num_ps > 2, "number of workers must be atleast 3 if evaluator role is required"

    global running
    if running:
        raise RuntimeError("An experiment is currently running. Please call experiment.end() to stop it.")

    try:
        global app_id
        global experiment_json
        global run_id
        running = True

        sc = util._find_spark().sparkContext
        app_id = str(sc.applicationId)

        versioned_path = _setup_experiment(versioned_resources, ps_impl._get_logdir(app_id, run_id), app_id, run_id)

        experiment_json = util._populate_experiment(sc, name, 'experiment', 'parameter_server', None, versioned_path, description)

        util._publish_experiment(app_id, run_id, experiment_json, 'CREATE')

        retval, logdir = ps_impl._launch(sc, map_fun, run_id, local_logdir=local_logdir, name=name, evaluator=evaluator)

        experiment_json = util._finalize_experiment(experiment_json, None, retval)

        util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')
    except:
        _exception_handler()
        raise
    finally:
        #cleanup spark jobs
        run_id +=1
        running = False
        sc.setJobGroup("", "")

    return logdir

def mirrored(map_fun, name='no-name', local_logdir=False, versioned_resources=None, description=None, evaluator=False):
    """
    *Distributed Training*

    Example usage:

    >>> from hops import experiment
    >>> def mirrored_training():
    >>>    import tensorflow
    >>>    from hops import tensorboard
    >>>    from hops import devices
    >>>    logdir = tensorboard.logdir()
    >>>    ...MirroredStrategy()...
    >>> experiment.mirrored(mirrored_training, local_logdir=True)

    Args:
        :map_fun: contains the code where you are using MirroredStrategy.
        :name: name of the experiment
        :local_logdir: True if *tensorboard.logdir()* should be in the local filesystem, otherwise it is in HDFS
        :versioned_resources: A list of HDFS paths of resources to version with this experiment
        :description: a longer description for the experiment

    Returns:
        HDFS path in your project where the experiment is stored

    """

    num_ps = util.num_param_servers()
    assert num_ps == 0, "number of parameter servers should be 0"

    global running
    if running:
        raise RuntimeError("An experiment is currently running. Please call experiment.end() to stop it.")

    num_workers = util.num_executors()
    if evaluator:
        assert num_workers > 2, "number of workers must be atleast 3 if evaluator role is required"

    try:
        global app_id
        global experiment_json
        global run_id
        running = True

        sc = util._find_spark().sparkContext
        app_id = str(sc.applicationId)

        mirrored_impl.run_id = run_id

        versioned_path = _setup_experiment(versioned_resources, mirrored_impl._get_logdir(app_id, run_id), app_id, run_id)

        experiment_json = util._populate_experiment(sc, name, 'experiment', 'mirrored', None, versioned_path, description)

        util._publish_experiment(app_id, run_id, experiment_json, 'CREATE')

        retval, logdir = mirrored_impl._launch(sc, map_fun, run_id, local_logdir=local_logdir, name=name, evaluator=evaluator)

        experiment_json = util._finalize_experiment(experiment_json, None, retval)

        util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')
    except:
        _exception_handler()
        raise
    finally:
        #cleanup spark jobs
        run_id +=1
        running = False
        sc.setJobGroup("", "")

    return logdir

def _exception_handler():
    """

    Returns:

    """
    global running
    global experiment_json
    if running and experiment_json != None:
        experiment_json = json.loads(experiment_json)
        experiment_json['status'] = "FAILED"
        experiment_json['finished'] = time.time()
        experiment_json = json.dumps(experiment_json)
        util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')

def _exit_handler():
    """

    Returns:

    """
    global running
    global experiment_json
    if running and experiment_json != None:
        experiment_json = json.loads(experiment_json)
        experiment_json['status'] = "KILLED"
        experiment_json['finished'] = time.time()
        experiment_json = json.dumps(experiment_json)
        util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')

atexit.register(_exit_handler)

def _setup_experiment(versioned_resources, logdir, app_id, run_id):
    versioned_path = util._version_resources(versioned_resources, logdir)
    hopshdfs.mkdir(hopshdfs.get_plain_path(util._get_experiments_dir()) + "/" + app_id + "_" + str(run_id))
    return versioned_path

def _finalize_experiment(experiment_json, hp, app_id,_run_id):
    experiment_json = util._finalize_experiment(experiment_json, hp, None)

    util._publish_experiment(app_id, run_id, experiment_json, 'REPLACE')
