from pathlib import Path
import copy

import spikeextractors as se
import spiketoolkit as st

from ..basesorter import BaseSorter

try:
    import herdingspikes as hs
    HAVE_HS = True
except ImportError:
    HAVE_HS = False


class HerdingspikesSorter(BaseSorter):
    """
    HerdingSpikes is a sorter based on estimated spike location, developed by
    researchers at the University of Edinburgh. It's a fast and scalable choice.

    See: HILGEN, Gerrit, et al. Unsupervised spike sorting for large-scale,
    high-density multielectrode arrays. Cell reports, 2017, 18.10: 2521-2532.
    """

    sorter_name = 'herdingspikes'
    installed = HAVE_HS
    requires_locations = True
    _default_params = None  # later

    installation_mesg = """
    More information on HerdingSpikes at:
      * https://github.com/mhhennig/hs2
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)
    
    @staticmethod
    def get_sorter_version():
        return hs.__version__

    def _setup_recording(self, recording, output_folder):
        
        p = self.params

        # Bandpass filter
        if p['filter'] and p['freq_min'] is not None and p['freq_max'] is not None:
            recording = st.preprocessing.bandpass_filter(
                recording=recording, freq_min=p['freq_min'], freq_max=p['freq_max'])

        if p['pre_scale']:
            recording = st.preprocessing.normalize_by_quantile(
                recording=recording, scale=p['pre_scale_value'],
                median=0.0, q1=0.05, q2=0.95
            )

        # this should have its name changed
        self.Probe = hs.probe.RecordingExtractor(
            recording,
            masked_channels=p['probe_masked_channels'],
            inner_radius=p['probe_inner_radius'],
            neighbor_radius=p['probe_neighbor_radius'],
            event_length=p['probe_event_length'],
            peak_jitter=p['probe_peak_jitter'])

    def _run(self, recording, output_folder):
        p = self.params

        if recording.is_filtered and p['filter']:
            print("Warning! The recording is already filtered, but Herding Spikes filter is enabled. You can disable "
                  "filters by setting 'filter' parameter to False")

        self.H = hs.HSDetection(
            self.Probe, file_directory_name=str(output_folder),
            left_cutout_time=p['left_cutout_time'],
            right_cutout_time=p['right_cutout_time'],
            threshold=p['detection_threshold'],
            to_localize=True,
            num_com_centers=p['num_com_centers'],
            maa=p['maa'],
            ahpthr=p['ahpthr'],
            out_file_name=p['out_file_name'],
            decay_filtering=p['decay_filtering'],
            save_all=p['save_all'],
            amp_evaluation_time=p['amp_evaluation_time'],
            spk_evaluation_time=p['spk_evaluation_time']
        )

        self.H.DetectFromRaw(load=True, tInc=100000)

        sorted_file = str(output_folder / 'HS2_sorted.hdf5')
        if(not self.H.spikes.empty):
            self.C = hs.HSClustering(self.H)
            self.C.ShapePCA(pca_ncomponents=p['pca_ncomponents'],
                            pca_whiten=p['pca_whiten'])
            self.C.CombinedClustering(
                alpha=p['clustering_alpha'],
                cluster_subset=p['clustering_subset'],
                bandwidth=p['clustering_bandwidth'],
                bin_seeding=p['clustering_bin_seeding'],
                n_jobs=p['clustering_n_jobs'],
                min_bin_freq=p['clustering_min_bin_freq']
            )
        else:
            self.C = hs.HSClustering(self.H)

        print('Saving to', sorted_file)
        self.C.SaveHDF5(sorted_file, sampling=self.Probe.fps)

    @staticmethod
    def get_result_from_folder(output_folder):
        return se.HS2SortingExtractor(file_path=Path(output_folder) / 'HS2_sorted.hdf5', load_unit_info=True)


HerdingspikesSorter._default_params = {
    # core params
    'clustering_bandwidth': 5.0,
    'clustering_alpha': 6.0,
    'clustering_n_jobs': -1,
    'clustering_bin_seeding': True,
    'clustering_min_bin_freq': 10,
    'clustering_subset': None,
    'left_cutout_time': 0.2,
    'right_cutout_time': 0.8,
    'detection_threshold': 15,

    # extra probe params
    'probe_masked_channels': [],
    'probe_inner_radius': 70,
    'probe_neighbor_radius': 90,
    'probe_event_length': 0.26,
    'probe_peak_jitter': 0.2,

    # extra detection params
    'num_com_centers': 1,
    'maa': 12,
    'ahpthr': 11,
    'out_file_name': "HS2_detected",
    'decay_filtering': False,
    'save_all': False,
    'amp_evaluation_time': 0.14,
    'spk_evaluation_time': 1.0,

    # extra pca params
    'pca_ncomponents': 2,
    'pca_whiten': True,

    # bandpass filter
    'freq_min': 300.0,
    'freq_max': 6000.0,
    'filter': True,

    # rescale traces
    'pre_scale': True,
    'pre_scale_value': 20.0

}
