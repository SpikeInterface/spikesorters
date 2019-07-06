from pathlib import Path
import os
import shutil
import numpy as np
import copy

from ..basesorter import BaseSorter
import spikeextractors as se

try:
    import tridesclous as tdc
    HAVE_TDC = True
except ImportError:
    HAVE_TDC = False


class TridesclousSorter(BaseSorter):
    """
    tridesclous is one of the more convinient, fast and elegant
    spike sorter.
    Everyone should test it.
    """

    sorter_name = 'tridesclous'
    installed = HAVE_TDC

    _default_params = {
        'highpass_freq': 400.,
        'lowpass_freq': 5000.,
        'peak_sign': '-',
        'relative_threshold': 5,
        'peak_span_ms': 0.3,
        'wf_left_ms': -2.0,
        'wf_right_ms':  3.0,
        'nb_max': 20000,
        'alien_value_threshold': None, # in benchmark there are no artifact
        'feature_method': 'auto',   #peak_max/global_pca/by_channel_pca
        'cluster_method': 'auto',  #sawchaincut/dbscan/kmeans
    }

    _extra_gui_params = [
        {'name': 'lowpass_freq', 'type': 'float', 'value': 400.0, 'default': 400.0, 'title': "Low-pass frequency"},
        {'name': 'highpass_freq', 'type': 'float', 'value': 5000.0, 'default': 5000.0, 'title': "High-pass frequency"},
        {'name': 'peak_sign', 'type': 'str', 'value': '-', 'default': '-', 'title': "Negative or positive peak sign"},
        {'name': 'relative_threshold', 'type': 'float', 'value': 5.0, 'default': 5.0, 'title': "Relative threshold for detection"},
        {'name': 'peak_span_ms', 'type': 'float', 'value': 0.3, 'default': 0.3, 'title': "Time span of peaks for detected events (ms)"},
        {'name': 'wf_left_ms', 'type': 'float', 'value': -2.0, 'default': -2.0, 'title': "Waveform length before peak (ms)"},
        {'name': 'wf_right_ms', 'type': 'float', 'value': 3.0, 'default': 3.0, 'title': "Waveform length after peak (ms)"},
        {'name': 'nb_max', 'type': 'int', 'value': 20000, 'default': 20000, 'title': "nb_max"},
        {'name': 'alien_value_threshold', 'type': 'float', 'value': None, 'default': None, 'title': "Threshold for artifacts"},
        {'name': 'feature_method', 'type': 'str', 'value': 'auto', 'default': 'auto', 'title': "Feature Extraction Method"},
        {'name': 'cluster_method', 'type': 'str', 'value': 'auto', 'default': 'auto', 'title': "Clustering Method"},
    ]

    _gui_params = copy.deepcopy(BaseSorter._gui_params)
    for param in _extra_gui_params:
        _gui_params.append(param)


    installation_mesg = """
       >>> pip install https://github.com/tridesclous/tridesclous/archive/master.zip

    More information on tridesclous at:
      * https://github.com/tridesclous/tridesclous
      * https://tridesclous.readthedocs.io
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def _setup_recording(self, recording, output_folder):
        # reset the output folder
        if output_folder.is_dir():
            shutil.rmtree(str(output_folder))
        os.makedirs(str(output_folder))

        # save prb file:
        probe_file = output_folder / 'probe.prb'
        se.save_probe_file(recording, probe_file, format='spyking_circus')

        # source file
        if isinstance(recording, se.BinDatRecordingExtractor) and recording._frame_first:
            # no need to copy
            raw_filename = recording._datfile
            dtype = recording._timeseries.dtype.str
            nb_chan = len(recording._channels)
            offset = recording._timeseries.offset
        else:
            if self.debug:
                print('Local copy of recording')
            # save binary file (chunk by hcunk) into a new file
            raw_filename = output_folder / 'raw_signals.raw'
            n_chan = recording.get_num_channels()
            chunksize = 2**24// n_chan
            se.write_binary_dat_format(recording, raw_filename, time_axis=0, dtype='float32', chunksize=chunksize)
            dtype='float32'
            offset = 0

        # initialize source and probe file
        tdc_dataio = tdc.DataIO(dirname=str(output_folder))
        nb_chan = recording.get_num_channels()

        tdc_dataio.set_data_source(type='RawData', filenames=[str(raw_filename)],
                                   dtype=dtype, sample_rate=recording.get_sampling_frequency(),
                                   total_channel=nb_chan, offset=offset)
        tdc_dataio.set_probe_file(str(probe_file))
        if self.debug:
            print(tdc_dataio)

    def _run(self, recording, output_folder):
        nb_chan = recording.get_num_channels()
        
        tdc_dataio = tdc.DataIO(dirname=str(output_folder))
        

        
        # make catalogue
        chan_grps = list(tdc_dataio.channel_groups.keys())
        for chan_grp in chan_grps:
            
            # parameters can change depending the group
            catalogue_nested_params = make_nested_tdc_params(tdc_dataio, chan_grp, **self.params)
            #~ print(catalogue_nested_params)
            
            peeler_params = tdc.get_auto_params_for_peelers(tdc_dataio, chan_grp)
            #~ print(peeler_params)
            
            # check params and OpenCL when many channels
            use_sparse_template = False
            use_opencl_with_sparse = False
            if nb_chan >64 and not peeler_params['use_sparse_template']:
                print('OpenCL is not available processing will be slow, try install it')
            
            cc = tdc.CatalogueConstructor(dataio=tdc_dataio, chan_grp=chan_grp)
            tdc.apply_all_catalogue_steps(cc, catalogue_nested_params, verbose=self.debug, )
            if self.debug:
                print(cc)
            cc.make_catalogue_for_peeler()

            # apply Peeler (template matching)
            initial_catalogue = tdc_dataio.load_catalogue(chan_grp=chan_grp)
            peeler = tdc.Peeler(tdc_dataio)
            peeler.change_params(catalogue=initial_catalogue, **peeler_params)
            peeler.run(duration=None, progressbar=self.debug)

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.TridesclousSortingExtractor(output_folder)
        return sorting


def make_nested_tdc_params(tdc_dataio, chan_grp,
        highpass_freq=400.,
        lowpass_freq=5000.,
        peak_sign='-',
        relative_threshold=5,
        peak_span_ms= 0.3,
        wf_left_ms= -2.0,
        wf_right_ms= 3.0,
        nb_max=20000,
        alien_value_threshold=None,
        feature_method='auto',
        cluster_method='auto'):
    
    params = tdc.get_auto_params_for_catalogue(tdc_dataio, chan_grp=chan_grp)
    
    params['preprocessor']['highpass_freq'] = highpass_freq
    params['preprocessor']['lowpass_freq'] = lowpass_freq
    
    params['peak_detector']['peak_sign'] = peak_sign
    params['peak_detector']['relative_threshold'] = relative_threshold
    params['peak_detector']['peak_span_ms'] = peak_span_ms

    params['extract_waveforms']['wf_left_ms'] = wf_left_ms
    params['extract_waveforms']['wf_right_ms'] = wf_right_ms
    params['extract_waveforms']['nb_max'] = nb_max
    
    params['clean_waveforms']['alien_value_threshold'] = alien_value_threshold
    
    
    
    if feature_method != 'auto':
        params['feature_method'] = feature_method
        params['feature_kargs'] = {}
    
    if cluster_method != 'auto':
        params['cluster_method'] = cluster_method
        params['cluster_kargs'] = {}
    
    return params
