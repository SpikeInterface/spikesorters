import copy
from pathlib import Path
import sys

import spikeextractors as se

from ..basesorter import BaseSorter
from ..utils.shellscript import ShellScript
from ..sorter_tools import _call_command

try:
    import klusta
    import klustakwik2

    HAVE_KLUSTA = True
except ImportError:
    HAVE_KLUSTA = False


class KlustaSorter(BaseSorter):
    """
    """

    sorter_name = 'klusta'
    installed = HAVE_KLUSTA
    requires_locations = False

    _default_params = {
        'probe_file': None,
        'adjacency_radius': None,
        'threshold_strong_std_factor': 5,
        'threshold_weak_std_factor': 2,
        'detect_sign': -1,
        'extract_s_before': 16,
        'extract_s_after': 32,
        'n_features_per_channel': 3,
        'pca_n_waveforms_max': 10000,
        'num_starting_clusters': 50,
    }

    _extra_gui_params = [
        {'name': 'adjacency_radius', 'type': 'float', 'value': None, 'default': None, 'title': "Adjacency radius (microns)"},
        {'name': 'threshold_strong_std_factor', 'type': 'float', 'value': 5.0, 'default': 5.0,
         'title': "Threshold strong std factor"},
        {'name': 'threshold_weak_std_factor', 'type': 'float', 'value': 2.0, 'default': 2.0,
         'title': "Threshold weak std factor"},
        {'name': 'detect_sign', 'type': 'int', 'value': -1, 'default': -1,
         'title': "Use -1, 0, or 1, depending on the sign of the spikes in the recording"},
        {'name': 'extract_s_before', 'type': 'int', 'value': 16, 'default': 16, 'title': "Frames to extract before"},
        {'name': 'extract_s_after', 'type': 'int', 'value': 32, 'default': 32, 'title': "Frames to extract after"},
        {'name': 'n_features_per_channel', 'type': 'int', 'value': 3, 'default': 3,
         'title': "Number of features per channel"},
        {'name': 'pca_n_waveforms_max', 'type': 'int', 'value': 10000, 'default': 10000,
         'title': "Max number of waveforms for PCA"},
        {'name': 'num_starting_clusters', 'type': 'int', 'value': 50, 'default': 50,
         'title': "Starting number of clusters"},
    ]

    sorter_gui_params = copy.deepcopy(BaseSorter.sorter_gui_params)
    for param in _extra_gui_params:
        sorter_gui_params.append(param)

    installation_mesg = """
       >>> pip install Cython h5py tqdm
       >>> pip install click klusta klustakwik2

    More information on klusta at:
      * https://github.com/kwikteam/phy"
      * https://github.com/kwikteam/klusta
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @staticmethod
    def get_sorter_version():
        return klusta.__version__

    def _setup_recording(self, recording, output_folder):
        source_dir = Path(__file__).parent

        # alias to params
        p = self.params

        experiment_name = output_folder / 'recording'

        # save prb file:
        if p['probe_file'] is None:
            p['probe_file'] = output_folder / 'probe.prb'
            recording.save_to_probe_file(p['probe_file'], format='klusta', radius=p['adjacency_radius'])

        # source file
        if isinstance(recording, se.BinDatRecordingExtractor) and recording._time_axis == 0 and \
                      recording._timeseries.offset == 0:
            # no need to copy
            raw_filename = str(Path(recording._datfile).resolve())
            dtype = recording._timeseries.dtype.str
        else:
            # save binary file (chunk by hcunk) into a new file
            raw_filename = output_folder / 'recording.dat'
            n_chan = recording.get_num_channels()
            chunk_size = 2 ** 24 // n_chan
            dtype = 'int16'
            recording.write_to_binary_dat_format(raw_filename, time_axis=0, dtype=dtype, chunk_size=chunk_size)

        if p['detect_sign'] < 0:
            detect_sign = 'negative'
        elif p['detect_sign'] > 0:
            detect_sign = 'positive'
        else:
            detect_sign = 'both'

        # set up klusta config file
        with (source_dir / 'config_default.prm').open('r') as f:
            klusta_config = f.readlines()

        # Note: should use format with dict approach here
        klusta_config = ''.join(klusta_config).format(experiment_name,
                                                      p['probe_file'], raw_filename,
                                                      float(recording.get_sampling_frequency()),
                                                      recording.get_num_channels(), "'{}'".format(dtype),
                                                      p['threshold_strong_std_factor'], p['threshold_weak_std_factor'],
                                                      "'" + detect_sign + "'",
                                                      p['extract_s_before'], p['extract_s_after'],
                                                      p['n_features_per_channel'],
                                                      p['pca_n_waveforms_max'], p['num_starting_clusters']
                                                      )

        with (output_folder / 'config.prm').open('w') as f:
            f.writelines(klusta_config)

    def _run(self, recording, output_folder):
        if 'win' in sys.platform:
            shell_cmd = '''
                        klusta --overwrite {klusta_config}
                    '''.format(klusta_config=output_folder / 'config.prm')
        else:
            shell_cmd = '''
                        #!/bin/bash
                        klusta {klusta_config} --overwrite
                    '''.format(klusta_config=output_folder / 'config.prm')

        shell_cmd = ShellScript(shell_cmd, keep_temp_files=True)
        shell_cmd.start()

        retcode = shell_cmd.wait()

        if retcode != 0:
            raise Exception('klusta returned a non-zero exit code')

        if not (output_folder / 'recording.kwik').is_file():
            raise Exception('Klusta did not run successfully')

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.KlustaSortingExtractor(file_or_folder_path=Path(output_folder) / 'recording.kwik')
        return sorting
