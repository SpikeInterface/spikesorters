import copy
from pathlib import Path
import sys

import spikeextractors as se

from ..basesorter import BaseSorter
from ..utils.shellscript import ShellScript
from ..sorter_tools import recover_recording

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
    
    requires_locations = False

    _default_params = {
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

    _params_description = {
        'adjacency_radius': "Radius in um to build channel neighborhood ",
        'threshold_strong_std_factor': "Strong threshold for spike detection",
        'threshold_weak_std_factor': "Weak threshold for spike detection",
        'detect_sign': "Use -1 (negative), 1 (positive) or 0 (both) depending "
                       "on the sign of the spikes in the recording",
        'extract_s_before': "Number of samples to cut out before the peak",
        'extract_s_after': "Number of samples to cut out after the peak",
        'n_features_per_channel': "Number of PCA features per channel",
        'pca_n_waveforms_max': "Maximum number of waveforms for PCA",
        'num_starting_clusters': "Number of initial clusters",
    }

    sorter_description = """Klusta is a density-based spike sorter that uses a masked EM approach for clustering.
    For more information see https://doi.org/10.1038/nn.4268"""

    installation_mesg = """\nTo use Klusta run:\n
       >>> pip install Cython h5py tqdm
       >>> pip install click klusta klustakwik2

    More information on klusta at:
      * https://github.com/kwikteam/phy"
      * https://github.com/kwikteam/klusta
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)
    
    @classmethod
    def is_installed(cls):
        return HAVE_KLUSTA
    
    @staticmethod
    def get_sorter_version():
        return klusta.__version__

    def _setup_recording(self, recording, output_folder):
        source_dir = Path(__file__).parent

        # alias to params
        p = self.params

        experiment_name = output_folder / 'recording'

        # save prb file 
        # note: only one group here, the split is done in basesorter
        probe_file = output_folder / 'probe.prb'
        recording.save_to_probe_file(probe_file, grouping_property=None,
                                     radius=p['adjacency_radius'])

        # source file
        if isinstance(recording, se.BinDatRecordingExtractor) and recording._time_axis == 0 and \
                      recording._timeseries.offset == 0:
            # no need to copy
            raw_filename = str(Path(recording._datfile).resolve())
            dtype = recording._timeseries.dtype.str
        else:
            # save binary file (chunk by hcunk) into a new file
            raw_filename = output_folder / 'recording.dat'
            dtype = 'int16'
            recording.write_to_binary_dat_format(raw_filename, time_axis=0, dtype=dtype, chunk_mb=500)

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
                                                      probe_file, raw_filename,
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
        recording = recover_recording(recording)
        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        klusta --overwrite {klusta_config}
                    '''.format(klusta_config=output_folder / 'config.prm')
        else:
            shell_cmd = '''
                        #!/bin/bash
                        klusta {klusta_config} --overwrite
                    '''.format(klusta_config=output_folder / 'config.prm')

        shell_script = ShellScript(shell_cmd, script_path=output_folder / f'run_{self.sorter_name}',
                                   log_path=output_folder / f'{self.sorter_name}.log', verbose=self.verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('klusta returned a non-zero exit code')

        if not (output_folder / 'recording.kwik').is_file():
            raise Exception('Klusta did not run successfully')

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.KlustaSortingExtractor(file_or_folder_path=Path(output_folder) / 'recording.kwik')
        return sorting
