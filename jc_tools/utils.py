
'''
Functions and decorators useful for data science.

TODO make utils folder and divide into generic functions operating on python builtins, ml-specific, etc.

TODO check if some of these don't already exist in some other python library:
     eg. decorators: https://wiki.python.org/moin/PythonDecoratorLibrary
'''

import os
import imp
import time
import glob
import random
import pickle
import hashlib
import subprocess
from itertools import tee
from itertools import chain
from functools import wraps
from itertools import islice
from itertools import groupby
from collections import Counter
from collections import OrderedDict
from collections import MutableMapping

from typing import Optional

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

import numpy as np


def flatten(iterable, exclude_strings=True):
    '''
    TODO check: I think this only flattens 1 level down
    '''
    if exclude_strings:
        iterable = ( [ element ] if isinstance(element, (str, bytes)) else element
                     for element in iterable )
    return list(chain.from_iterable(iterable))


def flatten_dict(dictionary):
    '''
    turn into list of tuples
    '''

    key_value_list = [ (key, value)
                       for key, values in dictionary.items()
                           for value in values ]

    return key_value_list


def flatten_keys(dictionary, parent_key='', sep='.'):
    '''
    Copied code from https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys/6027615#6027615
    '''

    items = []

    for key, value in dictionary.items():
        new_key = parent_key + sep + key if parent_key else key

        if isinstance(value, MutableMapping):
            items.extend(flatten_keys(value, new_key, sep=sep).items())

        else:
            items.append((new_key, value))

    return dict(items)


def group_by(iterable, key, keep=lambda x: x, ordered=False, sort_key=None):
    '''
    helper function to group list into groups according to a field
    '''

    ## /!\ needs to be sorted according to same key it will be grouped with
    iterable = sorted(iterable, key=key)
    grouped = groupby(iterable, key=key)
    grouped = [ (k, [ keep(el) for el in list(values) ]) for k, values in grouped ]

    if ordered:
        sort_key = sort_key or key

        grouped = sorted(grouped, key=sort_key)
        grouped = OrderedDict(grouped)

    else:
        grouped = { k: values for k, values in grouped }

    return grouped


def group_sequentially(iterable, group_size):
    '''s -> (s0, s1), (s2, s3), (s4, s5), ...'''
    return zip(*(iter(iterable),) * group_size)


def pairwise(iterable):
    return group_sequentially(iterable, 2)


def get_duplicates(iterable):
    return [ item for item, count in Counter(iterable).items() if count > 1 ]


def partition(iterable, condition=bool):
    '''
    Splits a list into 2 lists based on a condition
    '''
    a, b = tee((condition(item), item) for item in iterable)
    return ([ item for cond, item in a if not cond ],
            [ item for cond, item in b if cond ])


def sliding_window(sequence, n=2, fill_value_start=None, fill_value_end=None):
    '''
    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    '''

    ## TODO fix, only works with lists
    if fill_value_start is not None:
        sequence = [ fill_value_start, ] + sequence
    if fill_value_end is not None:
        sequence = sequence + [ fill_value_end, ]

    iterable = iter(sequence)
    result = tuple(islice(iterable, n))

    if len(result) == n:
        yield result

    for element in iterable:

        result = result[1:] + (element,)
        yield result


def choose_subsample_if_possible(iterable, amount):
    ''' '''

    try:
        return random.sample(iterable, amount)
    except ValueError:
        return iterable

def invert_mapping(mapping: dict) -> dict:
    '''
    /!\ assumption that values are unique ! no checks done.
    '''
    inverted = { value: key for key, values in mapping.items() for value in values }
    return inverted

### ML-specific processing


def onehot(array, num_classes):
    '''
    array: vector or 1D array of category numbers
    '''
    onehot_labels = np.eye(num_classes)[array]
    return onehot_labels


### id generators / reliability


def get_time_id(time_format='%y%m%d%H%M%S'):
    return time.strftime(time_format, time.localtime(time.time()))


def hash_string(string):
    ''' '''

    string = string.encode('utf-8')

    function_hash = hashlib.md5()
    function_hash.update(string)
    function_hash = function_hash.hexdigest()

    return function_hash

### decorators / practical / caching

class MacOSFile():
    '''
    from https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    '''

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

def import_module(module_filepath, function_name):
    '''
    '''

    dirpath, filename = os.path.split(module_filepath)
    module_name, fp_extension = filename.split('.')
    assert fp_extension == 'py'

    module_file, filepath, description = imp.find_module(module_name, [dirpath])
    module = imp.load_module(module_name, module_file, filepath, description)

    function = module.__dict__[function_name]

    return function


def measure_time(function):

    @wraps(function)
    def timed_function(*args, **kwargs):

        log.info('timing function %s ...' % function.__name__)
        started = time.time()
        result = function(*args, **kwargs)
        ended = time.time()
        duration = ended - started
        log.info('function %s took %2.2f sec to run' % (function.__name__, duration))
        ## TODO turn time into most logical human readable format

        return result

    return timed_function


def store_using_pickle(what, where):
    '''
    '''
    with open(where, 'wb') as ofile:
        pickle.dump(what, ofile)
    return

def read_using_pickle(where):
    '''
    '''

    with open(where, 'rb') as ifile:
        output = pickle.load(ifile)

    return output

def read_npy(where):
    arr = np.load(where)
    try:
        arr = arr.item()
    except ValueError:
        pass
    return arr

def store_npy(what, where):
    np.save(where, what)
    os.rename('{}.npy'.format(where), where)
    return


def store_output(storing_function=store_using_pickle, reading_function=read_using_pickle):
    '''
    TODO turn into Class based decorator bc too many parameters
         http://scottlobdell.me/2015/04/decorators-arguments-python/
    '''

    def decorator(function):
        '''
        Memoization decorator to store output of the function inside
        a chosen folder (eg. /tmp/function_outputs/)
        '''

        dirpath = '/tmp/tmp_function_outputs'
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

        stored_outputs_fps = glob.glob(os.path.join(dirpath, '*'))
        stored_outputs_fns = [ os.path.basename(fp) for fp in stored_outputs_fps ]

        stored_outputs = { func_hash: fp
            for func_hash, fp in zip(stored_outputs_fns, stored_outputs_fps) }

        @wraps(function)
        def wrapped_function(*args, **kwargs):
            '''
            /!\ TODO doesn't check hash of arguments, just str
            TODO also check hash of function code, but exclude comments and logging
            TODO capture time it took to compute output
            TODO make it work reliably with sets, dicts, etc.
            '''

            function_name = function.__name__
            function_args = str(args) + str(kwargs)

            string_to_hash = function_name + function_args
            function_hash = hash_string(string_to_hash)
            ## TODO doesn't work with lists, sets, dicts etc., without ordering them

            output_fp = stored_outputs.get(function_hash)
            output_is_stored = bool(output_fp)

            if output_is_stored:
                log.info('Fetching output of function %s from file at %s' % (function_name, output_fp))

                output = reading_function(where=output_fp)

            else:
                output_fp = os.path.join(dirpath, str(function_hash))
                log.info('Computing output of function %s and saving at %s' % (function_name, output_fp))
                output = function(*args, **kwargs)

                storing_function(what=output, where=output_fp)
                log.info('Saved output of function %s at %s' % (function_name, output_fp))

            return output

        return wrapped_function

    return decorator

storing_function = lambda what, where: pickle.dump(what, MacOSFile(open(where, 'wb')), protocol=pickle.HIGHEST_PROTOCOL)

reading_function = lambda where: pickle.load(MacOSFile(open(where, 'rb')))

def get_filepath(server_fp: str, to_path: Optional[str]=None, force_download: bool=False) -> str:
    '''
    Helper function to use a server file whether the code is running locally or
    on the server. Downloads it to temporary cache behind the scenes if run locally.

    :param server_fp: string, fullpath of file on the server
    :param to_path: string, fullpath of file locally, i.e. where to download it to
                    TODO this optional to_path suggests this function needs to be broken down

    TODO factor out server ?? what did I mean by this
    TODO proper error handling
    TODO use pathlib.Path ? https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists
    TODO move this to iproov.io ? I keep looking there instead
    TODO how to avoid file corruption ? (happens when stopped mid-download)
    '''

    if os.path.exists(server_fp):
        return server_fp


    if to_path:
        local_dir = os.path.dirname(to_path)

    else:
        original_filename = os.path.basename(server_fp)
        local_filename = '{hash_fp}_{fn}'.format(hash_fp=hash_string(server_fp), fn=original_filename)

        local_dir = '/tmp/cache/'
        to_path   = os.path.join(local_dir, local_filename)

    if os.path.exists(to_path) and not force_download:
        return to_path


    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    from_path = 'deep01:%s' % server_fp

    logging.info('downloading file locally from %s to %s ...' % (from_path, to_path))
    exit_code = subprocess.Popen(['scp', from_path, to_path]).wait()
    file_exists_on_server = not bool(exit_code)

    if not file_exists_on_server:
        raise FileNotFoundError('could not find file at %s' % from_path)

    return to_path


### experiment logging / etc. organisation


def log_repo_info(repo_path, show_changes=True):
    '''
    '''

    git_dir_args = [ '--git-dir %s/.git/' % repo_path,
                     '--work-tree=%s' % repo_path,
                     ]
    git_dir_arg = ' '.join(git_dir_args)

    command = 'git %s log -1 --format=%%H' % git_dir_arg
    exit_code, commit_hash = subprocess.getstatusoutput(command)
    git_repo_is_present = (exit_code == 0)

    if git_repo_is_present:
        log.info('commit hash: %s' % commit_hash)

        exit_code, diff_output = subprocess.getstatusoutput('git %s diff' % git_dir_arg)
        if exit_code == 0:
            if not diff_output:
                log.info('no changes in git repository')
            else:
                if show_changes:
                    log.warning('changes in git repository:\n%s' % diff_output)
                else:
                    log.error('changes in git repository')

    else:
        log.warning('not in a git repo, cannot track version')

    return


if __name__ == '__main__':

    from code import interact; interact(local=dict(globals(), **locals()))

