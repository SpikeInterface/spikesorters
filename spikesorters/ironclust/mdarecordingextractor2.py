import json
import numpy as np
import os

from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

from .mdaio import DiskReadMda, readmda, writemda32, writemda64, writemda


class MdaRecordingExtractor2(RecordingExtractor):
    def __init__(self, dataset_directory, *, raw_fname='raw.mda', params_fname='params.json'):
        RecordingExtractor.__init__(self)
        self._dataset_directory = dataset_directory
        if '/' not in raw_fname:
            # relative path
            self._timeseries_path = dataset_directory + '/' + raw_fname
        else:
            # absolute path
            self._timeseries_path = raw_fname
        self._dataset_params = read_dataset_params(dataset_directory, params_fname)
        self._samplerate = self._dataset_params['samplerate'] * 1.0

        geom0 = dataset_directory + '/geom.csv'
        self._geom_fname = ca.realizeFile(path=geom0)
        self._geom = np.genfromtxt(self._geom_fname, delimiter=',')

        timeseries_path = self._timeseries_path

        X = DiskReadMda(timeseries_path)
        if self._geom.shape[0] != X.N1():
            # raise Exception(
            #    'Incompatible dimensions between geom.csv and timeseries file {} <> {}'.format(self._geom.shape[0], X.N1()))
            print('WARNING: Incompatible dimensions between geom.csv and timeseries file {} <> {}'.format(self._geom.shape[0], X.N1()))
            self._geom = np.zeros((X.N1(), 2))

        self._num_channels = X.N1()
        self._num_timepoints = X.N2()
        for m in range(self._num_channels):
            self.set_channel_property(m, 'location', self._geom[m, :])

    def hash(self):
        from mountainclient import client as mt
        obj = dict(
            raw=mt.computeFileSha1(self._timeseries_path),
            geom=mt.computeFileSha1(self._geom_fname),
            params=self._dataset_params
        )
        return mt.sha1OfObject(obj)

    def recordingDirectory(self):
        return self._dataset_directory

    def get_channel_ids(self):
        return list(range(self._num_channels))

    def get_num_frames(self):
        return self._num_timepoints

    def get_sampling_frequency(self):
        return self._samplerate

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        X = DiskReadMda(self._timeseries_path)
        recordings = X.readChunk(i1=0, i2=start_frame, N1=X.N1(), N2=end_frame - start_frame)
        recordings = recordings[channel_ids, :]
        return recordings

    @staticmethod
    def write_recording(recording, save_path, params=dict(), raw_fname='raw.mda', params_fname='params.json', 
            _preserve_dtype=False):
        channel_ids = recording.get_channel_ids()
        M = len(channel_ids)
        # N = recording.get_num_frames()
        raw = recording.get_traces()
        location0 = recording.get_channel_property(channel_ids[0], 'location')
        nd = len(location0)
        geom = np.zeros((M, nd))
        for ii in range(len(channel_ids)):
            location_ii = recording.get_channel_property(channel_ids[ii], 'location')
            geom[ii, :] = list(location_ii)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        if _preserve_dtype:
            writemda(raw, save_path + '/' + raw_fname, dtype=raw.dtype)
        else:
            writemda32(raw, save_path + '/' + raw_fname)
        params["samplerate"] = recording.get_sampling_frequency()
        with open(save_path + '/' + params_fname, 'w') as f:
            json.dump(params, f)
        np.savetxt(save_path + '/geom.csv', geom, delimiter=',')


class SFMdaSortingExtractor(SortingExtractor):
    def __init__(self, firings_file):
        SortingExtractor.__init__(self)
        self._firings_path = firings_file

        self._firings = readmda(self._firings_path)
        self._times = self._firings[1, :]
        self._labels = self._firings[2, :]
        self._unit_ids = np.unique(self._labels).astype(int)

    def get_unit_ids(self):
        return self._unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        inds = np.where((self._labels == unit_id) & (start_frame <= self._times) & (self._times < end_frame))
        return np.rint(self._times[inds]).astype(int)

    def hash(self):
        from mountaintools import client as mt
        return mt.computeFileSha1(self._firings_path)

    @staticmethod
    def write_sorting(sorting, save_path):
        unit_ids = sorting.get_unit_ids()
        # if len(unit_ids) > 0:
        #     K = np.max(unit_ids)
        # else:
        #     K = 0
        times_list = []
        labels_list = []
        for i in range(len(unit_ids)):
            unit = unit_ids[i]
            times = sorting.get_unit_spike_train(unit_id=unit)
            times_list.append(times)
            labels_list.append(np.ones(times.shape) * unit)
        all_times = _concatenate(times_list)
        all_labels = _concatenate(labels_list)
        sort_inds = np.argsort(all_times)
        all_times = all_times[sort_inds]
        all_labels = all_labels[sort_inds]
        L = len(all_times)
        firings = np.zeros((3, L))
        firings[1, :] = all_times
        firings[2, :] = all_labels
        writemda64(firings, save_path)


def _concatenate(list):
    if len(list) == 0:
        return np.array([])
    return np.concatenate(list)


def read_dataset_params(dsdir, params_fname):
    fname1 = dsdir + '/' + params_fname
    if not os.path.exists(fname1):
        raise Exception('Dataset parameter file does not exist: ' + fname1)
    with open(fname1) as f:
        return json.load(f)
