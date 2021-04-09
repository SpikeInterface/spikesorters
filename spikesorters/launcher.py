"""
Utils functions to launch several sorter on several recording in parralell or not.
"""
import os
from pathlib import Path
import multiprocessing
import shutil
import json
import traceback
import json

import spikeextractors as se

from .sorterlist import sorter_dict, run_sorter


def _run_one(arg_list):
    # the multiprocessing python module force to have one unique tuple argument
    rec, sorter_name, output_folder, grouping_property, verbose, params, run_sorter_kwargs = arg_list
    if isinstance(rec, dict):
        recording = se.load_extractor_from_dict(rec)
    else:
        recording = rec

    SorterClass = sorter_dict[sorter_name]
    sorter = SorterClass(recording=recording, output_folder=output_folder,
                         grouping_property=grouping_property, verbose=verbose, delete_output_folder=False)
    sorter.set_params(**params)
    sorter.run(**run_sorter_kwargs)


def run_sorters(sorter_list, recording_dict_or_list, working_folder, sorter_params={}, grouping_property=None,
                mode='raise', engine=None, engine_kwargs={}, verbose=False, with_output=True, run_sorter_kwargs={}):
    """
    Run several sorters on several recordings.

    Parameters
    ----------
    sorter_list: list of str
        List of sorter names to run.
    recording_dict_or_list: dict or list
        A dict of recordings. The key will be the name of the recording.
        If a list is given then the name will be recording_0, recording_1, ...
    working_folder: str
        The working directory.
        This must not exist before calling this function.
    sorter_params: dict of dict with sorter_name as key
        This allow to overwrite default params for sorter.
    grouping_property: str or None
        The property of grouping given to sorters.
    mode: 'raise' or 'overwrite' or 'keep'
        The mode when the subfolder of recording/sorter already exists.
            * 'raise' : raise error if subfolder exists
            * 'overwrite' : force recompute
            * 'keep' : do not compute again if f=subfolder exists and log is OK
    engine: 'loop' or 'multiprocessing' or 'dask'
        Which approach to use to run the multiple sorters.
            * 'loop' : run sorters in a loop (serially)
            * 'multiprocessing' : use the Python multiprocessing library to run in parallel
            * 'dask' : use the Dask module to run in parallel
    engine_kwargs: dict
        This contains kwargs specific to the launcher engine:
            * 'loop' : no kargs
            * 'multiprocessing' : {'processes' : } number of processes
            * 'dask' : {'client':} the dask client for submiting task
    verbose: bool
        Controls sorter verbosity.
    with_output: bool
        return the output.
    run_sorter_kwargs: dict
        This contains kwargs specific to run_sorter function:\
            * 'raise_error' :  bool
            * 'parallel' : bool
            * 'n_jobs' : int
            * 'joblib_backend' : 'loky' / 'multiprocessing' / 'threading'

    Returns
    -------
    results : dict
        The output is nested dict[(rec_name, sorter_name)] of SortingExtractor.

    Notes
    -----
    Using multiprocessing through this function does not allow for subprocesses, so
    sorters that already use internally multiprocessing will fail.
    """
    working_folder = Path(working_folder)
    if mode == 'raise':
        assert not working_folder.is_dir(), "'working_folder' already exists, please remove it"

    if engine is None:
        engine = 'loop'

    for sorter_name in sorter_list:
        assert sorter_name in sorter_dict, '{} is not in sorter list'.format(sorter_name)

    if isinstance(recording_dict_or_list, list):
        # in case of list
        recording_dict = {'recording_{}'.format(i): rec for i, rec in enumerate(recording_dict_or_list)}
    elif isinstance(recording_dict_or_list, dict):
        recording_dict = recording_dict_or_list
    else:
        raise (ValueError('bad recording dict'))

    # when  grouping_property is not None : split in subrecording
    # but the subrecording must have len=1 because otherwise it break
    # the internal organisation of folder name.
    if grouping_property is not None:
        for rec_name, recording in recording_dict.items():
            recording_list = recording.get_sub_extractors_by_property(grouping_property)
            n_group = len(recording_list)
            assert n_group == 1, 'run_sorters() works only if grouping_property=None or if it split into one subrecording'
            recording_dict[rec_name] = recording_list[0]
        grouping_property = None

    need_serialize = engine != 'loop'

    task_list = []
    for rec_name, recording in recording_dict.items():
        for sorter_name in sorter_list:

            output_folder = working_folder / rec_name / sorter_name

            if is_log_ok(output_folder):
                # check is output_folders exists
                if mode == 'raise':
                    raise (Exception('output folder already exists for {} {}'.format(rec_name, sorter_name)))
                elif mode == 'overwrite':
                    shutil.rmtree(str(output_folder))
                elif mode == 'keep':
                    continue
                else:
                    raise (ValueError('mode not in raise, overwrite, keep'))
            params = sorter_params.get(sorter_name, {})
            if need_serialize:
                assert recording.check_if_dumpable(), 'run_sorters(engine=... ) if engine is not "loop" then recording have to be dumpable'
                rec = recording.dump_to_dict()
            else:
                rec = recording
            task_list.append((rec, sorter_name, output_folder, grouping_property, verbose, params, run_sorter_kwargs))

    if engine == 'loop':
        # simple loop in main process
        for arg_list in task_list:
            _run_one(arg_list)

    elif engine == 'multiprocessing':
        # use mp.Pool
        processes = engine_kwargs.get('processes', None)
        pool = multiprocessing.Pool(processes)
        pool.map(_run_one, task_list)
        pool.close()

    elif engine == 'dask':
        client = engine_kwargs.get('client', None)
        assert client is not None, 'For dask engine you have to provide : client = dask.distributed.Client(...)'

        tasks = []
        for arg_list in task_list:
            task = client.submit(_run_one, arg_list)
            tasks.append(task)

        for task in tasks:
            task.result()

    if with_output:
        if engine == 'dask':
            print('Warning!! With engine="dask" you cannot have directly output results\n' \
                  'Use : run_sorters(..., with_output=False)\n' \
                  'And then: results = collect_sorting_outputs(output_folders)')
            return

        results = collect_sorting_outputs(working_folder)
        return results


def is_log_ok(output_folder):
    # log is OK when run_time is not None
    if (output_folder / 'spikeinterface_log.json').is_file():
        with open(output_folder / 'spikeinterface_log.json', mode='r', encoding='utf8') as logfile:
            log = json.load(logfile)
            run_time = log.get('run_time', None)
            ok = run_time is not None
            return ok
    return False


def iter_output_folders(output_folders):
    output_folders = Path(output_folders)
    for rec_name in os.listdir(output_folders):
        if not (output_folders / rec_name).is_dir():
            continue
        for sorter_name in os.listdir(output_folders / rec_name):
            output_folder = output_folders / rec_name / sorter_name
            if not output_folder.is_dir():
                continue
            if not is_log_ok(output_folder):
                continue
            yield rec_name, sorter_name, output_folder


def iter_sorting_output(output_folders):
    """
    Iterator over output_folder to retrieve all triplets
    (rec_name, sorter_name, sorting)
    """
    for rec_name, sorter_name, output_folder in iter_output_folders(output_folders):
        SorterClass = sorter_dict[sorter_name]
        sorting = SorterClass.get_result_from_folder(output_folder)
        yield rec_name, sorter_name, sorting


def collect_sorting_outputs(output_folders):
    """
    Collect results in a output_folders.

    The output is a  dict with double key access results[(rec_name, sorter_name)] of SortingExtractor.
    """
    results = {}
    for rec_name, sorter_name, sorting in iter_sorting_output(output_folders):
        results[(rec_name, sorter_name)] = sorting
    return results
