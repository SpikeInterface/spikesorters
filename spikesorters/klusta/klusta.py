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
        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        klusta --overwrite {klusta_config}
                    '''.format(klusta_config=output_folder / 'config.prm')
        else:
            shell_cmd = '''
                        #!/bin/bash
                        klusta {klusta_config} --overwrite
                    '''.format(klusta_config=output_folder / 'config.prm')

        shell_script = ShellScript(shell_cmd, script_path=output_folder / f'run_{self.sorter_name}')
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
