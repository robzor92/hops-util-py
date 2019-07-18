"""
Utility functions for experiments
"""

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