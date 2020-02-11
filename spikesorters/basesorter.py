"""
Here a proposal for the futur Sorter with class approach.

The main idea is to decompose all intermediate steps to get more
flexibility:
  * setup the recording (traces, output folder, and so...)
  * set parameters
  * run the sorter (with futur possibility to make it in separate env/container)
  * get the result (SortingExtractor)

One benfit shoudl to compare the "run" time between sorter without
the setup and getting result.

One new idea usefull for tridesclous and maybe other sorter would
a way to adapt params with datasets.


"""

import time
import copy
from pathlib import Path
import threading
import shutil
import os
import datetime
import json
import traceback

import numpy as np

import spikeextractors as se


class BaseSorter:
    sorter_name = ''  # convinience for reporting
    installed = False  # check at class level if isntalled or not
    SortingExtractor_Class = None  # convinience to get the extractor
    requires_locations = False
    _default_params = {}
    sorter_gui_params = [
        {'name': 'output_folder', 'type': 'folder', 'value': None, 'default': None,
         'title': "Sorting output folder path", 'base_param': True},
        {'name': 'verbose', 'type': 'bool', 'value': True, 'default': True,
         'title': "The verbosity of the underlying spike sorter.", 'base_param': True},
        {'name': 'grouping_property', 'type': 'str', 'value': None, 'default': None,
         'title': "Will sort the recording by the given property ('group', etc.)", 'base_param': True},
        {'name': 'parallel', 'type': 'bool', 'value': False, 'default': False,
         'title': "If the recording is sorted by a property, then it will do this in parallel", 'base_param': True},
        {'name': 'delete_output_folder', 'type': 'bool', 'value': False, 'default': False,
         'title': "If True, delete the results of the sorter, otherwise, it won't.", 'base_param': True},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, recording=None, output_folder=None, verbose=False,
                 grouping_property=None, parallel=False, delete_output_folder=False):

        assert self.installed, """This sorter {} is not installed.
        Please install it with:  \n{} """.format(self.sorter_name, self.installation_mesg)
        if self.requires_locations:
            if 'location' not in recording.get_shared_channel_property_names():
                raise RuntimeError("Channel locations are required for this spike sorter. "
                                   "Locations can be added to the RecordingExtractor by loading a probe file "
                                   "(.prb or .csv) or by setting them manually.")

        self.verbose = verbose
        self.grouping_property = grouping_property
        self.parallel = parallel
        self.params = self.default_params()

        if output_folder is None:
            output_folder = 'tmp_' + self.sorter_name
        output_folder = Path(output_folder).absolute()

        if grouping_property is None:
            # only one groups
            self.recording_list = [recording]
            self.output_folders = [output_folder]
            if 'group' in recording.get_shared_channel_property_names():
                print("WARNING! The recording contains group. In order to spike sort by 'group' use "
                      "grouping_property='group' as argument.")
        else:
            # several groups
            if grouping_property not in recording.get_shared_channel_property_names():
                raise RuntimeError(f"'{grouping_property}' is not one of the channel properties.")
            self.recording_list = recording.get_sub_extractors_by_property(grouping_property)
            n_group = len(self.recording_list)
            self.output_folders = [output_folder / str(i) for i in range(n_group)]

        # make folders
        for output_folder in self.output_folders:
            if not output_folder.is_dir():
                os.makedirs(str(output_folder))
        self.delete_folders = delete_output_folder

    @classmethod
    def default_params(self):
        return copy.deepcopy(self._default_params)

    def set_params(self, **params):
        bad_params = []
        for p in params.keys():
            if p not in self._default_params.keys():
                bad_params.append(p)
        if len(bad_params) > 0:
            raise AttributeError('Bad parameters: ' + str(bad_params))
        self.params.update(params)
        
        # dump parameters inside the folder with json
        self._dump_params()
    
    def _dump_params(self):
        for output_folder in self.output_folders:
            with open(str(output_folder / 'spikeinterface_params.json'), 'w', encoding='utf8') as f:
                json.dump(_check_json(self.params), f, indent=4)
    
    def run(self, raise_error=True):
        for i, recording in enumerate(self.recording_list):
            self._setup_recording(recording, self.output_folders[i])
        
        # dump again params because some sorter do a folder reset (tdc)
        self._dump_params()
        
        now = datetime.datetime.now()
        
        log = {
            'sorter_name' : str(self.sorter_name),
            'sorter_version': str(self.get_sorter_version()),
            'datetime': now,
        }

        t0 = time.perf_counter()
        
        if raise_error:
            if not self.parallel:
                for i, recording in enumerate(self.recording_list):
                    self._run(recording, self.output_folders[i])
            else:
                # run in threads
                threads = []
                for i, recording in enumerate(self.recording_list):
                    thread = threading.Thread(target=self._run, args=(recording, self.output_folders[i]))
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
            t1 = time.perf_counter()
            run_time = float(t1-t0)
            
            log['error'] = False
            
        else:
            try:
                if not self.parallel:
                    for i, recording in enumerate(self.recording_list):
                        self._run(recording, self.output_folders[i])
                else:
                    # run in threads
                    threads = []
                    for i, recording in enumerate(self.recording_list):
                        thread = threading.Thread(target=self._run, args=(recording, self.output_folders[i]))
                        threads.append(thread)
                        thread.start()
                    for thread in threads:
                        thread.join()
                t1 = time.perf_counter()
                run_time = float(t1-t0)

            except Exception as err:
                run_time = None
                log['error'] = True
                log['error_trace'] = traceback.format_exc()
        
        log['run_time'] = run_time
        
        # dump log inside folders
        for output_folder in self.output_folders:
            with open(str(output_folder / 'spikeinterface_log.json'), 'w', encoding='utf8') as f:
                json.dump(_check_json(log), f, indent=4)

        if self.verbose:
            if run_time is None:
                print('Error running', self.sorter_name)
            else:
                print('{} run time {:0.2f}s'.format(self.sorter_name, t1 - t0))

        return run_time

    @staticmethod
    def get_sorter_version():
        # need be iplemented in subclass
        raise NotImplementedError

    def _setup_recording(self, recording, output_folder):
        # need be iplemented in subclass
        # this setup ONE recording (or SubExtractor)
        raise NotImplementedError

    def _run(self, recording, output_folder):
        # need be iplemented in subclass
        # this run the sorter on ONE recording (or SubExtractor)
        raise NotImplementedError

    @staticmethod
    def get_result_from_folder(output_folder):
        raise NotImplementedError

    def get_result_list(self):
        sorting_list = []
        for i, _ in enumerate(self.recording_list):
            sorting = self.get_result_from_folder(self.output_folders[i])
            sorting_list.append(sorting)
        return sorting_list

    def get_result(self):
        sorting_list = self.get_result_list()
        if len(sorting_list) == 1:
            sorting = sorting_list[0]
        else:
            for i, sorting in enumerate(sorting_list):
                group = self.recording_list[i].get_channel_property(self.recording_list[i].get_channel_ids()[0],
                                                                    'group')
                if sorting is not None:
                    for unit in sorting.get_unit_ids():
                        sorting.set_unit_property(unit, 'group', group)

            # reassemble the sorting outputs
            sorting_list = [sort for sort in sorting_list if sort is not None]
            multi_sorting = se.MultiSortingExtractor(sortings=sorting_list)
            sorting = multi_sorting

        if self.delete_folders:
            for out in self.output_folders:
                if self.verbose:
                    print("Removing ", str(out))
                shutil.rmtree(str(out), ignore_errors=True)
        sorting.set_sampling_frequency(self.recording_list[0].get_sampling_frequency())
        return sorting



def _check_json(d):
    # quick hack to ensure json writable
    
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
        elif isinstance(v, (np.int, np.int32, np.int64)):
            d[k] = int(v)
        elif isinstance(v,  (np.float, np.float32, np.float64)):
            d[k] = float(v)
        elif isinstance(v, datetime.datetime):
            d[k] = v.isoformat()

    return d
    
    