import copy
from pathlib import Path
import os
import sys
from typing import Union
import shutil
import numpy as np

import spikeextractors as se
from ..basesorter import BaseSorter
from ..sorter_tools import _call_command_split


def check_if_installed(kilosort_path: Union[str, None]):
    if kilosort_path is None:
        return False
    assert isinstance(kilosort_path, str)

    if kilosort_path.startswith('"'):
        kilosort_path = kilosort_path[1:-1]
    kilosort_path = str(Path(kilosort_path).absolute())

    if (Path(kilosort_path) / 'preprocessData.m').is_file():
        return True
    else:
        return False


class KilosortSorter(BaseSorter):
    """
    """

    sorter_name: str = 'kilosort'
    kilosort_path: Union[str, None] = os.getenv('KILOSORT_PATH', None)
    installed = check_if_installed(kilosort_path)

    _default_params = {
        'detect_threshold': 6,
        'car': True,
        'useGPU': True,
        'electrode_dimensions': None,
        'freq_min': 300,
        'freq_max': 6000
    }

    _extra_gui_params = [
        {'name': 'detect_threshold', 'type': 'float', 'value': 6.0, 'default': 6.0, 'title': "Relative detection threshold"},
        {'name': 'car', 'type': 'bool', 'value': True, 'default': True, 'title': "car"},
        {'name': 'useGPU', 'type': 'bool', 'value': True, 'default': True, 'title': "If True, will use GPU"},
        {'name': 'electrode_dimensions', 'type': 'list', 'value': None, 'default': None, 'title': "Electrode dimensions of probe"},
        {'name': 'freq_min', 'type': 'float', 'value': 300.0, 'default': 300.0, 'title': "Low-pass frequency"},
        {'name': 'freq_max', 'type': 'float', 'value': 6000.0, 'default': 6000.0, 'title': "High-pass frequency"},
    ]

    _gui_params = copy.deepcopy(BaseSorter._gui_params)
    for param in _extra_gui_params:
        _gui_params.append(param)

    installation_mesg = """\nTo use Kilosort run:\n
        >>> git clone https://github.com/cortex-lab/KiloSort
    and provide the installation path by setting the KILOSORT_PATH
    environment variables or using KilosortSorter.set_kilosort_path().\n\n

    More information on KiloSort at:
        https://github.com/cortex-lab/KiloSort
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @staticmethod
    def set_kilosort_path(kilosort_path: str):
        KilosortSorter.kilosort_path = kilosort_path
        KilosortSorter.installed = check_if_installed(KilosortSorter.kilosort_path)
        try:
            print("Setting KILOSORT_PATH environment variable for subprocess calls to:", kilosort_path)
            os.environ["KILOSORT_PATH"] = kilosort_path
        except Exception as e:
            print("Could not set KILOSORT_PATH environment variable:", e)

    def _setup_recording(self, recording, output_folder):
        source_dir = Path(__file__).parent
        p = self.params

        if not check_if_installed(KilosortSorter.kilosort_path):
            raise Exception(KilosortSorter.installation_mesg)
        assert isinstance(KilosortSorter.kilosort_path, str)

        # prepare electrode positions
        electrode_dimensions = p['electrode_dimensions']
        if electrode_dimensions is None:
            electrode_dimensions = [0, 1]
        if 'group' in recording.get_channel_property_names():
            groups = recording.get_channel_groups()
        else:
            groups = [1] * recording.get_num_channels()
        if 'location' in recording.get_channel_property_names():
            positions = np.array(recording.get_channel_locations())
        else:
            print("'location' information is not found. Using linear configuration")
            positions = np.array(
                [[0, i_ch] for i_ch in range(recording.get_num_channels())])
            electrode_dimensions = [0, 1]

        # save binary file
        input_file_path = output_folder / 'recording'
        se.write_binary_dat_format(recording, input_file_path, dtype='int16')

        # set up kilosort config files and run kilosort on data
        with (source_dir / 'kilosort_master.m').open('r') as f:
            kilosort_master_txt = f.read()
        with (source_dir / 'kilosort_config.m').open('r') as f:
            kilosort_config_txt = f.read()
        with (source_dir / 'kilosort_channelmap.m').open('r') as f:
            kilosort_channelmap_txt = f.read()

        nchan = recording.get_num_channels()
        Nfilt = (nchan // 32) * 32 * 8
        if Nfilt == 0:
            Nfilt = nchan * 8
        Nt = 128 * 1024 + 64

        if p['useGPU']:
            useGPU = 1
        else:
            useGPU = 0

        if p['car']:
            use_car = 1
        else:
            use_car = 0

        # make substitutions in txt files
        kilosort_master_txt = kilosort_master_txt.format(
            kilosort_path=str(
                Path(KilosortSorter.kilosort_path).absolute()),
            output_folder=str(output_folder),
            channel_path=str(
                (output_folder / 'kilosort_channelmap.m').absolute()),
            config_path=str((output_folder / 'kilosort_config.m').absolute()),
            useGPU=useGPU, 
        )
        
        kilosort_config_txt = kilosort_config_txt.format(
            nchanTOT=recording.get_num_channels(),
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            dat_file=str((output_folder / 'recording.dat').absolute()),
            Nfilt=Nfilt,
            Nt=Nt,
            kilo_thresh=p['detect_threshold'],
            use_car=use_car,
            freq_min=p['freq_min'],
            freq_max=p['freq_max']
        )

        kilosort_channelmap_txt = kilosort_channelmap_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            xcoords=list(positions[:, electrode_dimensions[0]]),
            ycoords=list(positions[:, electrode_dimensions[1]]),
            kcoords=groups
        )

        for fname, value in zip(['kilosort_master.m', 'kilosort_config.m',
                                 'kilosort_channelmap.m'],
                                [kilosort_master_txt, kilosort_config_txt,
                                 kilosort_channelmap_txt]):
            with (output_folder / fname).open('w') as f:
                f.writelines(value)

        shutil.copy(str(source_dir.parent / 'kilosort_npy_utils' / 'writeNPY.m'), str(output_folder))
        shutil.copy(str(source_dir.parent / 'kilosort_npy_utils' / 'constructNPYheader.m'), str(output_folder))

    def _run(self, recording, output_folder):
        cmd = "matlab -nosplash -nodisplay -r 'run {}; quit;'".format(output_folder / 'kilosort_master.m')
        if self.debug:
            print(cmd)
        if "win" in sys.platform:
            cmd_list = ['matlab', '-nosplash', '-nodisplay', '-wait',
                        '-r','run {}; quit;'.format(output_folder / 'kilosort_master.m')]
        else:
            cmd_list = ['matlab', '-nosplash', '-nodisplay',
                        '-r', 'run {}; quit;'.format(output_folder / 'kilosort_master.m')]

        # retcode = _run_command_and_print_output_split(cmd_list)
        _call_command_split(cmd_list)

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.KiloSortSortingExtractor(output_folder)
        return sorting
