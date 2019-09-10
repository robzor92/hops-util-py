"""

Models

"""

from hops import constants, util, hdfs
from hops.experiment_impl.util import experiment_utils
from hops.exceptions import RestAPIError
import json
import sys
import os
import time

import urllib3
urllib3.disable_warnings(urllib3.exceptions.SecurityWarning)

def get_best_model(name, metric, direction):

    if direction == Metric.MAX:
        direction = "desc"
    elif direction == Metric.MIN:
        direction = "asc"
    else:
        raise Exception("Invalid direction, should be Metric.MAX or Metric.MIN")

    headers = {constants.HTTP_CONFIG.HTTP_CONTENT_TYPE: constants.HTTP_CONFIG.HTTP_APPLICATION_JSON}

    resource_url = constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_REST_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_PROJECT_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   hdfs.project_id() + constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_MODELS_RESOURCE + \
                   "?filter_by=name_eq:" + name + "&sort_by=" + metric + ":" + direction + "&limit=1"

    response_object = util.send_request_with_session('GET', resource_url, headers=headers)

    if not response_object.ok:
        raise ModelNotFound("No model with name: could be found".format(name))

    print(resource_url)
    print(response_object)

    return json.loads(response_object.content.decode("UTF-8"))['items'][0]


def get_model(name, version):

    headers = {constants.HTTP_CONFIG.HTTP_CONTENT_TYPE: constants.HTTP_CONFIG.HTTP_APPLICATION_JSON}

    resource_url = constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_REST_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_PROJECT_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   hdfs.project_id() + constants.DELIMITERS.SLASH_DELIMITER + \
                   constants.REST_CONFIG.HOPSWORKS_MODELS_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
                   str(name) + "_" + str(version)

    print(resource_url)

    response_object = util.send_request_with_session('GET', resource_url, headers=headers)

    if response_object.ok:
        return response_object

    raise ModelNotFound("No model with name: {} and version {} could be found".format(name, version))


def export(model_path, model_name, model_version=None, overwrite=False, metrics=None, description=None, synchronous=True, synchronous_timeout=300):
    """
    Copies a trained model to the Models directory in the project and creates the directory structure of:

    >>> Models
    >>>      |
    >>>      - model_name
    >>>                 |
    >>>                 - version_x
    >>>                 |
    >>>                 - version_y

    For example if you run this:

    >>> from hops import model
    >>> model.export("iris_knn.pkl", "irisFlowerClassifier", 1, overwrite=True)

    it will copy the local model file "iris_knn.pkl" to /Projects/projectname/Models/irisFlowerClassifier/1/iris.knn.pkl
    on HDFS, and overwrite in case there already exists a file with the same name in the directory.

    If you run:

    >>> model.export("Resources/iris_knn.pkl", "irisFlowerClassifier", metrics={'accuracy': accuracy})

    If "model" is a directory on the local path exported by TensorFlow, and you run:
:
    >>> model.export("/model", "mnist", metrics={'accuracy': accuracy, 'loss': loss})

    It will copy the model directory contents to /Projects/projectname/Models/mnist/1/ , e.g the "model.pb" file and
    the "variables" directory.

    Args:
        :model_path: path to the trained model (HDFS or local)
        :model_name: name of the model
        :model_version: version of the model
        :overwrite: boolean flag whether to overwrite in case a model already exists in the exported directory
        :metrics: dict of evaluation metrics to attach to model
        :description: description about the model
        :synchronous: whether to synchronously wait for the model to be indexed in the models rest endpoint
        :synchronous_timeout: timeout in seconds for waiting for the model to be indexed

    Returns:
        The path to where the model was exported

    Raises:
        :ValueError: if there was an error with the exportation of the model due to invalid user input
    """

    if not description:
        description = 'A collection of models for ' + model_name

    # Make sure model name is a string, users could supply numbers
    model_name = str(model_name)

    project_path = hdfs.project_path()

    assert hdfs.exists(project_path + "Models"), "Your project is missing a dataset named Models, please create it."

    if not hdfs.exists(model_path) and not os.path.exists(model_path):
        raise ValueError("the provided model_path: {} , does not exist in HDFS or on the local filesystem".format(
            model_path))

    # parameters and metrics need to be dict, also no keys can be shared between them
    if metrics:
        _validate_metadata(metrics)

    model_dir_hdfs = project_path + constants.MODEL_SERVING.MODELS_DATASET + \
                     constants.DELIMITERS.SLASH_DELIMITER + model_name + constants.DELIMITERS.SLASH_DELIMITER

    # User did not specify model_version, pick the current highest version + 1, set to 1 if no model exists
    version_list = []
    if not model_version and hdfs.exists(model_dir_hdfs):
        model_version_directories = hdfs.ls(model_dir_hdfs)
        for version_dir in model_version_directories:
            try:
                if hdfs.isdir(version_dir):
                    version_list.append(int(version_dir[len(model_dir_hdfs):]))
            except:
                pass
        if len(version_list) > 0:
            model_version = max(version_list) + 1

    if not model_version:
        model_version = 1

    # Path to directory in HDFS to put the model files
    model_version_dir_hdfs = model_dir_hdfs + str(model_version)

    # If version directory already exists and we are not overwriting it then fail
    if not overwrite and hdfs.exists(model_version_dir_hdfs):
        raise ValueError("Could not create model directory: {}, the path already exists, "
                         "set flag overwrite=True "
                         "to remove the version directory and create the correct directory structure".format(model_version_dir_hdfs))

    # Overwrite version directory by deleting all content (this is needed for Provenance to register Model as deleted)
    if overwrite and hdfs.exists(model_version_dir_hdfs):
       hdfs.delete(model_version_dir_hdfs, recursive=True)
       hdfs.mkdir(model_version_dir_hdfs)

    # At this point we can create the version directory if it does not exists
    if not hdfs.exists(model_version_dir_hdfs):
       hdfs.mkdir(model_version_dir_hdfs)

    # Export the model files
    if os.path.exists(model_path):
        export_dir=_export_local_model(model_path, model_version_dir_hdfs, overwrite)
    else:
        export_dir=_export_hdfs_model(model_path, model_version_dir_hdfs, overwrite)

    # Attach modelName_modelVersion to experiment directory
    model_summary = {'name': model_name, 'version': model_version, 'metrics': experiment_utils._cast_keys_to_string(metrics), 'experimentId': None, 'description': description}
    if 'ML_ID' in os.environ:
        # Attach link from experiment to model
        experiment_utils._attach_model_link_xattr(os.environ['ML_ID'], model_name + '_' + str(model_version), 'CREATE')
        # Attach model metadata to models version folder
        model_summary['experimentId'] = os.environ['ML_ID']
        experiment_utils._attach_model_xattr(model_name + "_" + str(model_version), json.dumps(model_summary), 'CREATE')
    else:
        experiment_utils._attach_model_xattr(model_name + "_" + str(model_version), json.dumps(model_summary), 'CREATE')

    if synchronous:
        wait_interval = 86000

    # Model metadata is attached asynchronously by Epipe, therefore this necessary to ensure following steps in a pipeline will not fail
    if synchronous:
        start_time = time.time()
        for i in range(wait_interval):
            try:
                time.sleep(5)
                resp = get_model(model_name, model_version)
                if resp.ok:
                    return
                elif synchronous_timeout and ((time.time() - start_time) > synchronous_timeout):
                    return
            except ModelNotFound:
                pass

def _export_local_model(local_model_path, model_dir_hdfs, overwrite):
    """
    Exports a local directory of model files to Hopsworks "Models" dataset

     Args:
        :local_model_path: the path to the local model files
        :model_dir_hdfs: path to the directory in HDFS to put the model files
        :overwrite: boolean flag whether to overwrite existing model files

    Returns:
           the path to the exported model files in HDFS
    """
    if os.path.isdir(local_model_path):
        if not local_model_path.endswith(constants.DELIMITERS.SLASH_DELIMITER):
            local_model_path = local_model_path + constants.DELIMITERS.SLASH_DELIMITER
        for filename in os.listdir(local_model_path):
            hdfs.copy_to_hdfs(local_model_path + filename, model_dir_hdfs, overwrite=overwrite)

    if os.path.isfile(local_model_path):
        hdfs.copy_to_hdfs(local_model_path, model_dir_hdfs, overwrite=overwrite)

    return model_dir_hdfs


def _export_hdfs_model(hdfs_model_path, model_dir_hdfs, overwrite):
    """
    Exports a hdfs directory of model files to Hopsworks "Models" dataset

     Args:
        :hdfs_model_path: the path to the model files in hdfs
        :model_dir_hdfs: path to the directory in HDFS to put the model files
        :overwrite: boolean flag whether to overwrite in case a model already exists in the exported directory

    Returns:
           the path to the exported model files in HDFS
    """
    if hdfs.isdir(hdfs_model_path):
        for file_source_path in hdfs.ls(hdfs_model_path):
            model_name = file_source_path
            if constants.DELIMITERS.SLASH_DELIMITER in file_source_path:
                last_index = model_name.rfind(constants.DELIMITERS.SLASH_DELIMITER)
                model_name = model_name[last_index + 1:]
            dest_path = model_dir_hdfs + constants.DELIMITERS.SLASH_DELIMITER + model_name
            hdfs.cp(file_source_path, dest_path, overwrite=overwrite)
    elif hdfs.isfile(hdfs_model_path):
        model_name = hdfs_model_path
        if constants.DELIMITERS.SLASH_DELIMITER in hdfs_model_path:
            last_index = model_name.rfind(constants.DELIMITERS.SLASH_DELIMITER)
            model_name = model_name[last_index + 1:]
        dest_path = model_dir_hdfs + constants.DELIMITERS.SLASH_DELIMITER + model_name
        hdfs.cp(hdfs_model_path, dest_path, overwrite=overwrite)

    return model_dir_hdfs

def _validate_metadata(metrics):
    assert type(metrics) is dict, 'provided metrics is not in a dict'

class Metric:
    MAX = "MAX"
    MIN = "MIN"

class ModelNotFound(Exception):
    """This exception will be raised if the requested serving could not be found"""
