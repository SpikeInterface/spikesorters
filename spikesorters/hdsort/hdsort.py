from pathlib import Path
import os
from typing import Union
import sys

import spikeextractors as se
from ..basesorter import BaseSorter
from ..utils.shellscript import ShellScript
from ..sorter_tools import recover_recording


def check_if_installed(hdsort_path: Union[str, None]):
    if hdsort_path is None:
        return False
    assert isinstance(hdsort_path, str)

    if hdsort_path.startswith('"'):
        hdsort_path = hdsort_path[1:-1]
    hdsort_path = str(Path(hdsort_path).absolute())
    if (Path(hdsort_path) / '+hdsort').is_dir():
        return True
    else:
        return False


class HDSortSorter(BaseSorter):
    """
    """

    sorter_name: str = 'hdsort'
    hdsort_path: Union[str, None] = os.getenv('HDSORT_PATH', None)
    requires_locations = False
    _default_params = {
        'detect_threshold': 4.2,
        'detect_sign': -1,  # -1 - 1
        'filter': True,
        'parfor': True,
        'freq_min': 300,
        'freq_max': 7000,
        'max_el_per_group': 9,
        'min_el_per_group': 1,
        'add_if_nearer_than': 20,
        'max_distance_within_group': 52,
        'n_pc_dims': 6,
        'chunk_size': 500000,
        'loop_mode': 'local_parfor',
    }

    _params_description = {
        'detect_threshold': "Threshold for spike detection",
        'detect_sign': "Use -1 (negative) or 1 (positive) depending "
                       "on the sign of the spikes in the recording",
        'filter': "Enable or disable filter",
        'parfor': "If True, the Matlab parfor is used",
        'freq_min': "High-pass filter cutoff frequency",
        'freq_max': "Low-pass filter cutoff frequency",
        'max_el_per_group': "Maximum number of channels per electrode group",
        'min_el_per_group': "Minimum number of channels per electrode group",
        'add_if_nearer_than': "Minimum distance to add electrode to an electrode group",
        'max_distance_within_group': "Maximum distance within an electrode group",
        'n_pc_dims': "Number of principal components dimensions to perform initial clustering",
        'chunk_size': "Chunk size in number of frames for template-matching",
        'loop_mode': "Loop mode: 'loop', 'local_parfor', 'grid' (requires a grid architecture)",
    }

    sorter_description = """HDSort is a template-matching spike sorter designed for high density micro-electrode arrays. 
    For more information see https://doi.org/10.1152/jn.00803.2017"""

    installation_mesg = """\nTo use HDSort run:\n
        >>> git clone https://git.bsse.ethz.ch/hima_public/HDsort.git
    and provide the installation path by setting the HDSORT_PATH
    environment variables or using HDSortSorter.set_hdsort_path().\n\n

    More information on HDSort at:
        https://git.bsse.ethz.ch/hima_public/HDsort.git
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)
    
    @classmethod
    def is_installed(cls):
        return check_if_installed(cls.hdsort_path)
    
    @staticmethod
    def get_sorter_version():
        p = os.getenv('HDSORT_PATH', None)
        if p is None:
            return 'unknown'
        else:
            with open(os.path.join(p, 'version.txt'), mode='r', encoding='utf8') as f:
                version = f.readline()
        return version

    @staticmethod
    def set_hdsort_path(hdsort_path: str):
        HDSortSorter.hdsort_path = str(Path(hdsort_path).absolute())
        try:
            print("Setting HDSORT_PATH environment variable for subprocess calls to:", hdsort_path)
            os.environ["HDSORT_PATH"] = hdsort_path
        except Exception as e:
            print("Could not set HDSORT_PATH environment variable:", e)

    def _setup_recording(self, recording, output_folder):
        if not self.is_installed():
            raise Exception(HDSortSorter.installation_mesg)

        source_dir = Path(__file__).parent
        utils_path = source_dir.parent / 'utils'

        if isinstance(recording, se.Mea1kRecordingExtractor):
            self.params['file_name'] = str(Path(recording._file_path).absolute())
            self.params['file_format'] = 'mea1k'
            print('Using Mea1k format')
        elif isinstance(recording, se.MaxOneRecordingExtractor):
            self.params['file_name'] = str(Path(recording._file_path).absolute())
            self.params['file_format'] = 'maxone'
            print('Using MaxOne format')
        else:
            file_name = output_folder / 'recording.h5'
            # Generate three files dataset in Mea1k format
            se.Mea1kRecordingExtractor.write_recording(recording=recording, save_path=str(file_name))
            self.params['file_name'] = str(file_name.absolute())
            self.params['file_format'] = 'mea1k'

        p = self.params
        p['sort_name'] = 'hdsort_output'

        # read the template txt files
        with (source_dir / 'hdsort_master.m').open('r') as f:
            hdsort_master_txt = f.read()
        with (source_dir / 'hdsort_config.m').open('r') as f:
            hdsort_config_txt = f.read()

        # make substitutions in txt files
        hdsort_master_txt = hdsort_master_txt.format(
            hdsort_path=str(
                Path(HDSortSorter.hdsort_path).absolute()),
            utils_path=str(utils_path.absolute()),
            config_path=str((output_folder / 'hdsort_config.m').absolute()),
            file_name=p['file_name'],
            file_format=p['file_format'],
            sort_name=p['sort_name'],
            chunk_size=p['chunk_size'],
            loop_mode=p['loop_mode']
        )

        if p['filter']:
            p['filter'] = 1
        else:
            p['filter'] = 0

        if p['parfor']:
            p['parfor'] = 'true'
        else:
            p['parfor'] = 'false'

        hdsort_config_txt = hdsort_config_txt.format(
            filter=p['filter'],
            parfor=p['parfor'],
            hpf=p['freq_min'],
            lpf=p['freq_max'],
            max_el_per_group=p['max_el_per_group'],
            min_el_per_group=p['min_el_per_group'],
            add_if_nearer_than=p['add_if_nearer_than'],
            max_distance_within_group=p['max_distance_within_group'],
            detect_threshold=p['detect_threshold'],
            n_pc_dims=p['n_pc_dims'],
        )

        for fname, txt in zip(['hdsort_master.m', 'hdsort_config.m'],
                              [hdsort_master_txt, hdsort_config_txt]):
            with (output_folder / fname).open('w') as f:
                f.write(txt)

    def _run(self, recording, output_folder):
        recording = recover_recording(recording)
        tmpdir = output_folder
        os.makedirs(str(tmpdir), exist_ok=True)
        samplerate = recording.get_sampling_frequency()

        if recording.is_filtered and self.params['filter']:
            print("Warning! The recording is already filtered, but HDsort filter is enabled. You can disable "
                  "filters by setting 'filter' parameter to False")

        if "win" in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        cd {tmpdir}
                        matlab -nosplash -wait -r hdsort_master
                    '''.format(tmpdir=output_folder)
        else:
            shell_cmd = '''
                        #!/bin/bash
                        cd "{tmpdir}"
                        matlab -nosplash -nodisplay -r hdsort_master
                    '''.format(tmpdir=output_folder)

        shell_script = ShellScript(shell_cmd, script_path=output_folder / f'run_{self.sorter_name}',
                                   log_path=output_folder / f'{self.sorter_name}.log', verbose=self.verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('HDsort returned a non-zero exit code')

        samplerate_fname = str(output_folder / 'samplerate.txt')
        with open(samplerate_fname, 'w') as f:
            f.write('{}'.format(samplerate))

    @staticmethod
    def get_result_from_folder(output_folder):
        output_folder = Path(output_folder)
        sorting = se.HDSortSortingExtractor(file_path=str(output_folder / 'hdsort_output' /
                                                          'hdsort_output_results.mat'))

        return sorting
