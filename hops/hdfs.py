"""
API for interacting with the file system on Hops (HopsFS).

It is a wrapper around pydoop together with utility functions that are Hops-specific.
"""
import pydoop.hdfs as hdfs
import datetime
from six import string_types
import shutil
import stat as local_stat
import fnmatch
import os
import errno
import pydoop.hdfs.path as path
from hops import constants
import sys
import subprocess

fd = None

def project_id():
    """
    Get the Hopsworks project id from environment variables

    Returns: the Hopsworks project id

    """
    return os.environ[constants.ENV_VARIABLES.HOPSWORKS_PROJECT_ID_ENV_VAR]


def project_user():
    """
    Gets the project username ("project__user") from environment variables

    Returns:
        the project username
    """

    try:
        hops_user = os.environ[constants.ENV_VARIABLES.HADOOP_USER_NAME_ENV_VAR]
    except:
        hops_user = os.environ[constants.ENV_VARIABLES.HDFS_USER_ENV_VAR]
    return hops_user

def project_name():
    """
    Extracts the project name from the project username ("project__user")

    Returns:
        project name
    """
    hops_user = project_user()
    hops_user_split = hops_user.split("__")  # project users have username project__user
    project = hops_user_split[0]
    return project

def project_path(project=None):
    """ Get the path in HopsFS where the HopsWorks project is located. To point to a particular dataset, this path should be
    appended with the name of your dataset.

    >>> from hops import hdfs
    >>> project_path = hdfs.project_path()
    >>> print("Project path: {}".format(project_path))

    Args:
        :project_name: If this value is not specified, it will get the path to your project. If you need to path to another project, you can specify the name of the project as a string.

    Returns:
        returns the project absolute path
    """

    if project:
        # abspath means "hdfs://namenode:port/ is preprended
        return hdfs.path.abspath("/Projects/" + project + "/")
    project = project_name()
    return hdfs.path.abspath("/Projects/" + project + "/")

def get():
    """ Get a handle to pydoop hdfs using the default namenode (specified in hadoop config)

    Returns:
        Pydoop hdfs handle
    """
    return hdfs.hdfs('default', 0, user=project_user())


def get_fs():
    """ Get a handle to pydoop fs using the default namenode (specified in hadoop config)

    Returns:
        Pydoop fs handle
    """
    return hdfs.fs.hdfs('default', 0, user=project_user())


def _expand_path(hdfs_path, project="", exists=True):
    """
    Expands a given path. If the path is /Projects.. hdfs:// is prepended.
    If the path is ../ the full project path is prepended.

    Args:
        :hdfs_path the path to be expanded
        :exists boolean flag, if this is true an exception is thrown if the expanded path does not exist.

    Raises:
        IOError if exists flag is true and the path does not exist

    Returns:
        path expanded with HDFS and project
    """
    if project == "":
        project = project_name()
    # Check if a full path is supplied. If not, assume it is a relative path for this project - then build its full path and return it.
    if hdfs_path.startswith("/Projects/") or hdfs_path.startswith("/Projects"):
        hdfs_path = "hdfs://" + hdfs_path
    elif not hdfs_path.startswith("hdfs://"):
        # if the file URL type is not HDFS, throw an error
        if "://" in hdfs_path:
            raise IOError("path %s must be a full hdfs path or a relative path" % hdfs_path)
        proj_path = project_path(project)
        hdfs_path = proj_path + hdfs_path
    if exists == True and not hdfs.path.exists(hdfs_path):
        raise IOError("path %s not found" % hdfs_path)
    return hdfs_path


def _init_logger():
    """
    Initialize the logger by opening the log file and pointing the global fd to the open file
    """
    logfile = os.environ['EXEC_LOGFILE']
    fs_handle = get_fs()
    global fd
    try:
        fd = fs_handle.open_file(logfile, mode='w')
    except:
        fd = fs_handle.open_file(logfile, flags='w')


def log(string):
    """
    Logs a string to the log file

    Args:
        :string: string to log
    """
    global fd
    if fd:
        if isinstance(string, string_types):
            fd.write(('{0}: {1}'.format(datetime.datetime.now().isoformat(), string) + '\n').encode())
        else:
            fd.write(('{0}: {1}'.format(datetime.datetime.now().isoformat(),
                                        'ERROR! Attempting to write a non-string object to logfile') + '\n').encode())


def _kill_logger():
    """
    Closes the logfile
    """
    global fd
    if fd:
        try:
            fd.flush()
            fd.close()
        except:
            pass


def _create_directories(app_id, run_id, param_string, sub_type=None):
    """
    Creates directories for an experiment, if Experiments folder exists it will create directories
    below it, otherwise it will create them in the Logs directory.

    Args:
        :app_id: YARN application ID of the experiment
        :run_id: Experiment ID
        :param_string: name of the new directory created under parent directories
        :type: type of the new directory parent, e.g differential_evolution
        :sub_type: type of sub directory to parent, e.g generation

    Returns:
        The new directories for the yarn-application and for the execution (hdfs_exec_logdir, hdfs_appid_logdir)
    """

    pyhdfs_handle = get()


    experiments_dir = project_path() + "Experiments"

    # user may have accidently deleted Experiments dataset
    if not hdfs.exists(experiments_dir):
        hdfs.mkdir(experiments_dir)
        try:
            st = hdfs.stat(experiments_dir)
            if not bool(st.st_mode & local_stat.S_IWGRP):  # if not group writable make it so
                hdfs.chmod(experiments_dir, "g+w")
        except IOError:
            pass

    experiment_run_dir = experiments_dir + "/" + app_id + "_" + str(run_id)

    # determine directory structure based on arguments
    if sub_type:
        hdfs_exec_logdir = experiment_run_dir + "/" + str(sub_type) + '/' + str(param_string)
    elif not param_string and not sub_type:
        hdfs_exec_logdir = experiment_run_dir + '/'
    else:
        hdfs_exec_logdir = experiment_run_dir + '/' + str(param_string)

    # Need to remove directory if it exists (might be a task retry)
    if pyhdfs_handle.exists(hdfs_exec_logdir):
        pyhdfs_handle.delete(hdfs_exec_logdir, recursive=True)

    # create the new directory
    pyhdfs_handle.create_directory(hdfs_exec_logdir)

    # update logfile
    logfile = hdfs_exec_logdir + '/' + 'logfile'
    os.environ['EXEC_LOGFILE'] = logfile

    return experiment_run_dir


def copy_to_hdfs(local_path, relative_hdfs_path, overwrite=False, project=None):
    """
    Copies a path from local filesystem to HDFS project (recursively) using relative path in $CWD to a path in hdfs (hdfs_path)

    For example, if you execute:

    >>> copy_to_hdfs("data.tfrecords", "/Resources", project="demo")

    This will copy the file data.tfrecords to hdfs://Projects/demo/Resources/data.tfrecords

    Args:
        :local_path: Absolute or local path on the local filesystem to copy
        :relative_hdfs_path: a path in HDFS relative to the project root to where the local path should be written
        :overwrite: a boolean flag whether to overwrite if the path already exists in HDFS
        :project: name of the project, defaults to the current HDFS user's project
    """
    if project == None:
        project = project_name()

    if "PDIR" in os.environ:
        full_local = os.environ['PDIR'] + '/' + local_path
    else:
        # Absolute path
        if local_path.startswith(os.getcwd()):
            full_local = local_path
        else:
            # Relative path
            full_local = os.getcwd() + '/' + local_path

    hdfs_path = _expand_path(relative_hdfs_path, project, exists=False)

    if overwrite:
        hdfs_handle = get()
        # check if project path exist, if so delete it (since overwrite flag was set to true)
        hdfs_path = hdfs_path + "/" + os.path.basename(full_local)
        if hdfs_handle.exists(hdfs_path):
            hdfs_handle.delete(hdfs_path, recursive=True)

    print("Started copying " + hdfs_path + " on hdfs to path " + hdfs_path + "\n")

    # copy directories from local path to HDFS project path
    hdfs.put(full_local, hdfs_path)

    print("Finished copying\n")


def copy_to_local(hdfs_path, local_path="", overwrite=False, project=None):
    """
    Copies a directory or file from a HDFS project to a local private scratch directory. If there is not enough space on the local scratch directory, an exception is thrown.
    If the local file exists, and the hdfs file and the local file are the same size in bytes, return 'ok' immediately.
    If the local directory tree exists, and the hdfs subdirectory and the local subdirectory have the same files and directories, return 'ok' immediately.

    For example, if you execute:

    >>> copy_to_local("Resources/my_data")

    This will copy the directory my_data from the Resources dataset in your project to the current working directory on the path ./my_data

    Raises:
      IOError if there is not enough space to localize the file/directory in HDFS to the scratch directory ($PDIR)

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :local_path: the relative or full path to a directory on the local filesystem to copy to (relative to a scratch directory $PDIR), defaults to $CWD
        :overwrite: a boolean flag whether to overwrite if the path already exists in the local scratch directory.
        :project: name of the project, defaults to the current HDFS user's project

    Returns:
        the full local pathname of the file/dir
    """

    if project == None:
        project = project_name()

    if "PDIR" in os.environ:
        local_dir = os.environ['PDIR'] + '/' + local_path
    else:
        # Absolute path
        if local_path.startswith(os.getcwd()):
            local_dir = local_path
        else:
            # Relative path
            local_dir = os.getcwd() + '/' + local_path

    if not os.path.isdir(local_dir):
        raise IOError("You need to supply the path to a local directory. This is not a local dir: %s" % local_dir)

    filename = path.basename(hdfs_path)
    full_local = local_dir + "/" + filename

    project_hdfs_path = _expand_path(hdfs_path, project=project)

    # Get the amount of free space on the local drive
    stat = os.statvfs(local_dir)
    free_space_bytes = stat.f_bsize * stat.f_bavail

    hdfs_size = path.getsize(project_hdfs_path)

    if os.path.isfile(full_local) and not overwrite:
        sz = os.path.getsize(full_local)
        if hdfs_size == sz:
            print("File " + project_hdfs_path + " is already localized, skipping download...")
            return full_local
        else:
            os.remove(full_local)

    if os.path.isdir(full_local) and not overwrite:
        try:
            localized = _is_same_directory(full_local, project_hdfs_path)
            if localized:
                print("Full directory subtree already on local disk and unchanged. Set overwrite=True to force download")
                return full_local
            else:
                shutil.rmtree(full_local)
        except Exception as e:
            print("Failed while checking directory structure to avoid re-downloading dataset, falling back to downloading")
            print(e)
            shutil.rmtree(full_local)

    if hdfs_size > free_space_bytes:
        raise IOError("Not enough local free space available on scratch directory: %s" % local_path)

    if overwrite:
        if os.path.isdir(full_local):
            shutil.rmtree(full_local)
        elif os.path.isfile(full_local):
            os.remove(full_local)

    print("Started copying " + project_hdfs_path + " to local disk on path " + local_dir + "\n")

    hdfs.get(project_hdfs_path, local_dir)

    print("Finished copying\n")

    return full_local


def _is_same_directory(local_path, hdfs_path):
    """
    Validates that the same occurrence and names of files exists in both hdfs and local
    """
    local_file_list = []
    for root, dirnames, filenames in os.walk(local_path):
        for filename in fnmatch.filter(filenames, '*'):
            local_file_list.append(filename)
        for dirname in fnmatch.filter(dirnames, '*'):
            local_file_list.append(dirname)
    local_file_list.sort()

    hdfs_file_list = glob(hdfs_path + '/*', recursive=True)
    hdfs_file_list = [path.basename(str(r)) for r in hdfs_file_list]
    hdfs_file_list.sort()

    if local_file_list == hdfs_file_list:
        return True
    else:
        return False

def cp(src_hdfs_path, dest_hdfs_path):
    """
    Copy the contents of src_hdfs_path to dest_hdfs_path.

    If src_hdfs_path is a directory, its contents will be copied recursively. Source file(s) are opened for reading and copies are opened for writing.

    Args:
        :src_hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :dest_hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).

    """
    src_hdfs_path = _expand_path(src_hdfs_path)
    dest_hdfs_path = _expand_path(dest_hdfs_path)
    hdfs.cp(src_hdfs_path, dest_hdfs_path)


def _get_experiments_dir():
    """
    Gets the folder where the experiments are writing their results

    Returns:
        the folder where the experiments are writing results
    """
    pyhdfs_handle = get()
    if pyhdfs_handle.exists(project_path() + "Experiments"):
        return project_path() + "Experiments"
    elif pyhdfs_handle.exists(project_path() + "Logs"):
        return project_path() + "Logs/TensorFlow"


def glob(hdfs_path, recursive=False, project=None):
    """
    Finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order.

    Globbing gives you the list of files in a dir that matches a supplied pattern

    >>> glob('Resources/*.json')
    >>> ['Resources/1.json', 'Resources/2.json']

    glob is implemented as  os.listdir() and fnmatch.fnmatch()
    We implement glob as hdfs.ls() and fnmatch.filter()

    Args:
     :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to project_name in HDFS.
     :project: If the supplied hdfs_path is a relative path, it will look for that file in this project's subdir in HDFS.

    Raises:
        IOError if the supplied hdfs path does not exist

    Returns:
      A possibly-empty list of path names that match pathname, which must be a string containing a path specification. pathname can be either absolute
    """

    # Get the full path to the dir for the input glob pattern
    # "hdfs://Projects/jim/blah/*.jpg" => "hdfs://Projects/jim/blah"
    # Then, ls on 'hdfs://Projects/jim/blah', then filter out results
    if project == None:
        project = project_name()
    lastSep = hdfs_path.rfind("/")
    inputDir = hdfs_path[:lastSep]
    inputDir = _expand_path(inputDir, project)
    pattern = hdfs_path[lastSep + 1:]
    if not hdfs.path.exists(inputDir):
        raise IOError("Glob path %s not found" % inputDir)
    dirContents = hdfs.ls(inputDir, recursive=recursive)
    return fnmatch.filter(dirContents, pattern)


def ls(hdfs_path, recursive=False, project=None):
    """
    Returns all the pathnames in the supplied directory.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to project_name in HDFS).
        :recursive: if it is a directory and recursive is True, the list contains one item for every file or directory in the tree rooted at hdfs_path.
        :project: If the supplied hdfs_path is a relative path, it will look for that file in this project's subdir in HDFS.

    Returns:
      A possibly-empty list of path names stored in the supplied path.
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return hdfs.ls(hdfs_path, recursive=recursive)


def lsl(hdfs_path, recursive=False, project=None):
    """
    Returns all the pathnames in the supplied directory.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to project_name in HDFS).
        :recursive: if it is a directory and recursive is True, the list contains one item for every file or directory in the tree rooted at hdfs_path.
        :project: If the supplied hdfs_path is a relative path, it will look for that file in this project's subdir in HDFS.

    Returns:
        A possibly-empty list of path names stored in the supplied path.
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return hdfs.lsl(hdfs_path, recursive=recursive)


def rmr(hdfs_path, project=None):
    """
    Recursively remove files and directories.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to project_name in HDFS).
        :project: If the supplied hdfs_path is a relative path, it will look for that file in this project's subdir in HDFS.

    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return hdfs.rmr(hdfs_path)


def mkdir(hdfs_path, project=None):
    """
    Create a directory and its parents as needed.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to project_name in HDFS).
        :project: If the supplied hdfs_path is a relative path, it will look for that file in this project's subdir in HDFS.

    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project, exists=False)
    return hdfs.mkdir(hdfs_path)


def move(src, dest):
    """
    Move or rename src to dest.

    Args:
        :src: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :dest: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).

    """
    src = _expand_path(src, project_name())
    dest = _expand_path(dest, project_name(), exists=False)
    return hdfs.move(src, dest)


def rename(src, dest):
    """
    Rename src to dest.

    Args:
        :src: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :dest: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
    """
    src = _expand_path(src, project_name())
    dest = _expand_path(dest, project_name(), exists=False)
    return hdfs.rename(src, dest)


def chown(hdfs_path, user, group, project=None):
    """
    Change file owner and group.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to the given project path in HDFS).
        :user: New hdfs username
        :group: New hdfs group
        :project: If this value is not specified, it will get the path to your project. If you need to path to another project, you can specify the name of the project as a string.
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return hdfs.chown(hdfs_path, user, group)


def chmod(hdfs_path, mode, project=None):
    """
    Change file mode bits.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :mode: File mode (user/group/world privilege) bits
        :project: If this value is not specified, it will get the path to your project. If you need to path to another project, you can specify the name of the project as a string.
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return hdfs.chmod(hdfs_path, mode)


def stat(hdfs_path, project=None):
    """
    Performs the equivalent of os.stat() on path, returning a StatResult object.

    Args:
        :hdfs_path: If this value is not specified, it will get the path to your project. You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :project: If this value is not specified, it will get the path to your project. If you need to path to another project, you can specify the name of the project as a string.

    Returns:
        StatResult object
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return hdfs.stat(hdfs_path)


def access(hdfs_path, mode, project=None):
    """
    Perform the equivalent of os.access() on path.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :mode: File mode (user/group/world privilege) bits
        :project: If this value is not specified, it will get the path to your project. If you need to path to another project, you can specify the name of the project as a string.

    Returns:
        True if access is allowed, False if not.
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return hdfs.access(hdfs_path, mode)


def _mkdir_p(path):
    """
    Creates path on local filesystem

    Args:
        path to create

    Raises:
        OSError
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def open_file(hdfs_path, project=None, flags='rw', buff_size=0):
    """
    Opens an HDFS file for read/write/append and returns a file descriptor object (fd) that should be closed when no longer needed.

    Args:
        hdfs_path: you can specify either a full hdfs pathname or a relative one (relative to your project's path in HDFS)
        flags: supported opening modes are 'r', 'w', 'a'. In addition, a trailing 't' can be added to specify text mode (e.g, 'rt' = open for reading text)
        buff_size: Pass 0 as buff_size if you want to use the "configured" values, i.e the ones set in the Hadoop configuration files.

    Returns:
        A file descriptor (fd) that needs to be closed (fd-close()) when it is no longer needed.

    Raises:
        IOError: If the file does not exist.
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project, exists=False)
    fs_handle = get_fs()
    fd = fs_handle.open_file(hdfs_path, flags, buff_size=buff_size)
    return fd


def close():
    """
    Closes an the HDFS connection (disconnects to the namenode)
    """
    hdfs.close()


def exists(hdfs_path, project=None):
    """
    Return True if hdfs_path exists in the default HDFS.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :project: If this value is not specified, it will get the path to your project. If you need to path to another project, you can specify the name of the project as a string.


    Returns:
        True if hdfs_path exists.

    Raises: IOError
    """
    if project == None:
        project = project_name()

    try:
        hdfs_path = _expand_path(hdfs_path, project)
    except IOError:
        return False
    return hdfs.path.exists(hdfs_path)


def isdir(hdfs_path, project=None):
    """
    Return True if path refers to a directory.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :project: If this value is not specified, it will get the path to your project. If you need to path to another project, you can specify the name of the project as a string.

    Returns:
        True if path refers to a directory.

    Raises: IOError
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return hdfs.isdir(hdfs_path)


def isfile(hdfs_path, project=None):
    """
    Return True if path refers to a file.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :project: If this value is not specified, it will get the path to your project. If you need to path to another project, you can specify the name of the project as a string.

    Returns:
        True if path refers to a file.

    Raises: IOError
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return path.isfile(hdfs_path)

def isdir(hdfs_path, project=None):
    """
    Return True if path refers to a directory.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
        :project: If this value is not specified, it will get the path to your project. If you need to path to another project, you can specify the name of the project as a string.

    Returns:
        True if path refers to a file.

    Raises: IOError
    """
    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)
    return path.isdir(hdfs_path)


def capacity():
    """
    Returns the raw capacity of the filesystem

    Returns:
        filesystem capacity (int)
    """
    return hdfs.capacity()


def dump(data, hdfs_path):
    """
    Dumps data to a file

    Args:
        :data: data to write to hdfs_path
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).
    """

    hdfs_path = _expand_path(hdfs_path, exists=False)
    return hdfs.dump(data, hdfs_path)


def load(hdfs_path):
    """
    Read the content of hdfs_path and return it.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).

    Returns:
        the read contents of hdfs_path
    """
    hdfs_path = _expand_path(hdfs_path)
    return hdfs.load(hdfs_path)

def ls(hdfs_path, recursive=False):
    """
    lists a directory in HDFS

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).

    Returns:
        returns a list of hdfs paths
    """
    hdfs_path = _expand_path(hdfs_path)
    return hdfs.ls(hdfs_path, recursive=recursive)

def stat(hdfs_path):
    """
    Performs the equivalent of os.stat() on hdfs_path, returning a StatResult object.

    Args:
        :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).

    Returns:
        returns a list of hdfs paths
    """
    hdfs_path = _expand_path(hdfs_path)
    return hdfs.stat(hdfs_path)

def abs_path(hdfs_path):
    """
     Return an absolute path for hdfs_path.

     Args:
         :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS).

    Returns:
        Return an absolute path for hdfs_path.
    """
    return _expand_path(hdfs_path)

def add_module(hdfs_path, project=None):
    """
     Add a .py or .ipynb file from HDFS to sys.path

     For example, if you execute:

     >>> add_module("Resources/my_module.py")
     >>> add_module("Resources/my_notebook.ipynb")

     You can import it simply as:

     >>> import my_module
     >>> import my_notebook

     Args:
         :hdfs_path: You can specify either a full hdfs pathname or a relative one (relative to your Project's path in HDFS) to a .py or .ipynb file

     Returns:
        Return full local path to localized python file or converted python file in case of .ipynb file
    """

    localized_deps = os.getcwd() + "/localized_deps"
    if not os.path.exists(localized_deps):
        os.mkdir(localized_deps)
        open(localized_deps + '/__init__.py', mode='w').close()

    if localized_deps not in sys.path:
        sys.path.append(localized_deps)

    if project == None:
        project = project_name()
    hdfs_path = _expand_path(hdfs_path, project)

    if path.isfile(hdfs_path) and hdfs_path.endswith('.py'):
        py_path = copy_to_local(hdfs_path, localized_deps)
        if py_path not in sys.path:
            sys.path.append(py_path)
        return py_path
    elif path.isfile(hdfs_path) and hdfs_path.endswith('.ipynb'):
        ipynb_path = copy_to_local(hdfs_path, localized_deps)
        python_path = os.environ['PYSPARK_PYTHON']
        jupyter_binary = os.path.dirname(python_path) + '/jupyter'
        if not os.path.exists(jupyter_binary):
            raise Exception('Could not find jupyter binary on path {}'.format(jupyter_binary))

        converted_py_path = os.path.splitext(ipynb_path)[0] + '.py'
        if os.path.exists(converted_py_path):
            os.remove(converted_py_path)

        conversion = subprocess.Popen([jupyter_binary, 'nbconvert', '--to', 'python', ipynb_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = conversion.communicate()
        if conversion.returncode != 0:
            raise Exception("Notebook conversion to .py failed: stdout: {} \n stderr: {}".format(out, err))

        if not os.path.exists(converted_py_path):
            raise Exception('Could not find converted .py file on path {}'.format(converted_py_path))
        if converted_py_path not in sys.path:
            sys.path.append(converted_py_path)
        return converted_py_path
    else:
        raise Exception("Given path " + hdfs_path + " does not point to a .py or .ipynb file")



