import os
import shutil
import time

import pytest
import spikeextractors as se

from spikesorters import run_sorters, collect_sorting_outputs


def test_run_sorters_with_list():
    rec0, _ = se.example_datasets.toy_example(num_channels=4, duration=30, seed=0)
    rec1, _ = se.example_datasets.toy_example(num_channels=8, duration=30, seed=0)

    recording_list = [rec0, rec1]
    sorter_list = ['tridesclous']
    working_folder = 'test_run_sorters_list'
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)
    run_sorters(sorter_list, recording_list, working_folder, verbose=False)


def test_run_sorters_with_dict():
    rec0, _ = se.example_datasets.toy_example(num_channels=4, duration=30, seed=0)
    rec1, _ = se.example_datasets.toy_example(num_channels=8, duration=30, seed=0)

    recording_dict = {'toy_tetrode': rec0, 'toy_octotrode': rec1}

    # sorter_list = ['mountainsort4', 'klusta', 'tridesclous']
    # ~ sorter_list = ['tridesclous',  'klusta',]
    # ~ sorter_list = ['tridesclous', 'mountainsort4']
    sorter_list = ['tridesclous', 'herdingspikes']

    working_folder = 'test_run_sorters_dict'
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)

    sorter_params = {
        'tridesclous': dict(detect_threshold=5.6),
        'herdingspikes': dict(detect_threshold=20.1),
    }

    # simple loop
    t0 = time.perf_counter()
    results = run_sorters(sorter_list, recording_dict, working_folder, sorter_params=sorter_params, engine=None)
    t1 = time.perf_counter()
    print(t1 - t0)
    print(results)

    shutil.rmtree(working_folder + '/toy_tetrode/tridesclous')
    results = run_sorters(sorter_list, recording_dict, working_folder, engine=None, sorter_params=sorter_params, mode='keep')

@pytest.mark.skipif(True, reason='This bug with pytest/travis but not run directly')
def test_run_sorters_multiprocessing():
    recording_dict = {}
    for i in range(8):
        rec, _ = se.example_datasets.toy_example(num_channels=8, duration=30, seed=0, dumpable=True)
        recording_dict['rec_' + str(i)] = rec

    # sorter_list = ['mountainsort4', 'klusta', 'tridesclous']
    sorter_list = ['tridesclous', 'klusta', ]
    # ~ sorter_list = ['tridesclous', 'herdingspikes']

    working_folder = 'test_run_sorters_mp'
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)

    # multiprocessing
    t0 = time.perf_counter()
    run_sorters(sorter_list, recording_dict, working_folder, engine='multiprocessing', engine_kwargs={'processes': 4})
    t1 = time.perf_counter()
    print(t1 - t0)

@pytest.mark.skipif(True, reason='This bug with pytest/travis but not run directly')
def test_run_sorters_dask():
    # create a dask Client for a slurm queue
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    python = '/home/samuel.garcia/.virtualenvs/py36/bin/python3.6'
    cluster = SLURMCluster(processes=1, cores=1, memory="12GB", python=python, walltime='12:00:00',)
    cluster.scale(5)
    client = Client(cluster)
    
    # create recording
    recording_dict = {}
    for i in range(8):
        rec, _ = se.example_datasets.toy_example(num_channels=8, duration=30, seed=0, dumpable=True)
        recording_dict['rec_' + str(i)] = rec

    # sorter_list = ['mountainsort4', 'klusta', 'tridesclous']
    sorter_list = ['tridesclous', ]
    # ~ sorter_list = ['tridesclous', 'herdingspikes']

    working_folder = 'test_run_sorters_dask'
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)

    # dask
    t0 = time.perf_counter()
    results = run_sorters(sorter_list, recording_dict, working_folder, engine='dask',
                          engine_kwargs={'client': client}, with_output=True)
    # dask do not return results always None
    assert results is None
    t1 = time.perf_counter()
    print(t1 - t0)


def test_collect_sorting_outputs():
    working_folder = 'test_run_sorters_dict'
    results = collect_sorting_outputs(working_folder)
    print(results)


if __name__ == '__main__':
    test_run_sorters_with_list()

    test_run_sorters_with_dict()

    # test_run_sorters_multiprocessing()
    
    # test_run_sorters_dask()

    #~ test_collect_sorting_outputs()
