import copy
from pathlib import Path

import spikeextractors as se
from spiketoolkit.preprocessing import bandpass_filter, whiten

from ..basesorter import BaseSorter

try:
    import ml_ms4alg

    HAVE_MS4 = True
except ImportError:
    HAVE_MS4 = False


class Mountainsort4Sorter(BaseSorter):
    """
    Mountainsort
    """

    sorter_name = 'mountainsort4'
    installed = HAVE_MS4
    requires_locations = False

    _default_params = {
        'detect_sign': -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        'adjacency_radius': -1,  # Use -1 to include all channels in every neighborhood
        'freq_min': 300,  # Use None for no bandpass filtering
        'freq_max': 6000,
        'filter': True,
        'whiten': True,  # Whether to do channel whitening as part of preprocessing
        'curation': False,
        'num_workers': None,
        'clip_size': 50,
        'detect_threshold': 3,
        'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel
        'noise_overlap_threshold': 0.15,  # Use None for no automated curation'
    }

    installation_mesg = """
       >>> pip install ml_ms4alg

    More information on mountainsort at:
      * https://github.com/flatironinstitute/mountainsort
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @staticmethod
    def get_sorter_version():
        if hasattr(ml_ms4alg, '__version__'):
            return ml_ms4alg.__version__
        return 'unknown'

    def _setup_recording(self, recording, output_folder):
        pass

    def _run(self, recording, output_folder):
        
        # Sort
        # alias to params
        p = self.params

        if recording.is_filtered and p['filter']:
            print("Warning! The recording is already filtered, but Mountainsort4 filter is enabled. You can disable "
                  "filters by setting 'filter' parameter to False")

        samplerate = recording.get_sampling_frequency()

        # Bandpass filter
        if p['filter'] and p['freq_min'] is not None and p['freq_max'] is not None:
            recording = bandpass_filter(recording=recording, freq_min=p['freq_min'], freq_max=p['freq_max'])

        # Whiten
        if p['whiten']:
            recording = whiten(recording=recording)

        # Check location no more needed done in basesorter

        sorting = ml_ms4alg.mountainsort4(
            recording=recording,
            detect_sign=p['detect_sign'],
            adjacency_radius=p['adjacency_radius'],
            clip_size=p['clip_size'],
            detect_threshold=p['detect_threshold'],
            detect_interval=p['detect_interval'],
            num_workers=p['num_workers'],
            verbose=self.verbose
        )

        # Curate
        if p['noise_overlap_threshold'] is not None and p['curation'] is True:
            if self.verbose:
                print('Curating')
            sorting = ml_ms4alg.mountainsort4_curation(
                recording=recording,
                sorting=sorting,
                noise_overlap_threshold=p['noise_overlap_threshold']
            )

        se.MdaSortingExtractor.write_sorting(sorting, str(output_folder / 'firings.mda'))

        samplerate_fname = str(output_folder / 'samplerate.txt')
        with open(samplerate_fname, 'w') as f:
            f.write('{}'.format(samplerate))

    @staticmethod
    def get_result_from_folder(output_folder):
        output_folder = Path(output_folder)
        tmpdir = output_folder

        result_fname = str(tmpdir / 'firings.mda')
        samplerate_fname = str(tmpdir / 'samplerate.txt')
        with open(samplerate_fname, 'r') as f:
            samplerate = float(f.read())

        sorting = se.MdaSortingExtractor(file_path=result_fname, sampling_frequency=samplerate)
        return sorting
