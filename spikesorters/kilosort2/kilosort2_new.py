from pathlib import Path
import os
import sys
import numpy as np
from typing import Union

import spikeextractors as se
from ..basesorter import BaseSorter
from ..sorter_tools import _call_command_split


def check_if_installed(kilosort2_path: Union[str, None]):
    if kilosort2_path is None:
        return False
    assert isinstance(kilosort2_path, str)

    if kilosort2_path.startswith('"'):
        kilosort2_path = kilosort2_path[1:-1]
    kilosort2_path = str(Path(kilosort2_path).absolute())

    if (Path(kilosort2_path) / 'master_kilosort.m').is_file():
        return True
    else:
        return False


class Kilosort2SorterNew(BaseSorter):
    """
    """

    sorter_name: str = 'kilosort2_new'
    kilosort2_path: Union[str, None] = os.getenv('KILOSORT2_PATH', None)
    installed = check_if_installed(kilosort2_path)

    _default_params = {
        'detect_threshold': 5,
        'electrode_dimensions': None,
        'car': True,
        'minFR': 0.1,
    }

    installation_mesg = """\nTo use Kilosort run:\n
        >>> git clone https://github.com/cortex-lab/KiloSort
    and provide the installation path by setting the KILOSORT2_PATH
    environment variables or using Kilosort2SorterNew.set_kilosort2_path().\n\n

    More information on KiloSort2 at:
        https://github.com/MouseLand/Kilosort2
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @staticmethod
    def set_kilosort2_path(kilosort2_path: str):
        Kilosort2SorterNew.kilosort2_path = kilosort2_path

    def set_params(self, **params):
        BaseSorter.set_params(self, **params)

    def _setup_recording(self, recording, output_folder):

        source_dir = Path(Path(__file__).parent)

        p = self.params

        if not check_if_installed(Kilosort2SorterNew.kilosort2_path):
            raise Exception(Kilosort2SorterNew.installation_mesg)
        assert isinstance(Kilosort2SorterNew.kilosort2_path, str)

        # prepare electrode positions
        electrode_dimensions = p['electrode_dimensions']
        if electrode_dimensions is None:
            electrode_dimensions = [0, 1]
        if 'group' in recording.get_channel_property_names():
            groups = [recording.get_channel_property(
                ch, 'group') for ch in recording.get_channel_ids()]
        else:
            groups = [1] * recording.get_num_channels()
        if 'location' in recording.get_channel_property_names():
            positions = np.array([recording.get_channel_property(
                chan, 'location') for chan in recording.get_channel_ids()])
        else:
            print("'location' information is not found. Using linear configuration")
            positions = np.array(
                [[0, i_ch] for i_ch in range(recording.get_num_channels())])
            electrode_dimensions = [0, 1]

        # save binary file
        input_file_path = output_folder / 'recording'
        se.write_binary_dat_format(recording, input_file_path, dtype='int16')

        # read the template txt files
        with (source_dir / 'kilosort2_master_new.m').open('r') as f:
            kilosort2_master_txt = f.read()
        with (source_dir / 'kilosort2_config_new.m').open('r') as f:
            kilosort2_config_txt = f.read()
        with (source_dir / 'kilosort2_channelmap_new.m').open('r') as f:
            kilosort2_channelmap_txt = f.read()

        # make substitutions in txt files
        kilosort2_master_txt = kilosort2_master_txt.format(
            kilosort2_path=str(
                Path(Kilosort2SorterNew.kilosort2_path).absolute()),
            output_folder=str(output_folder),
            channel_path=str(
                (output_folder / 'kilosort2_channelmap.m').absolute()),
            config_path=str((output_folder / 'kilosort2_config.m').absolute()),
            sample_rate=recording.get_sampling_frequency()
        )

        if p['car']:
            use_car = 1
        else:
            use_car = 0

        kilosort2_config_txt = kilosort2_config_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            dat_file=str((output_folder / 'recording.dat').absolute()),
            minFR=p['minFR'],
            kilo_thresh=p['detect_threshold'],
            use_car=use_car
        )

        kilosort2_channelmap_txt = kilosort2_channelmap_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            xcoords=list(positions[:, electrode_dimensions[0]]),
            ycoords=list(positions[:, electrode_dimensions[1]]),
            kcoords=groups
        )

        for fname, txt in zip(['kilosort2_master.m', 'kilosort2_config.m',
                               'kilosort2_channelmap.m'],
                              [kilosort2_master_txt, kilosort2_config_txt,
                               kilosort2_channelmap_txt]):
            with (output_folder / fname).open('w') as f:
                f.write(txt)

    def _run(self, recording, output_folder):
        cmd = "matlab -nosplash -nodisplay -r 'run {}; quit;'".format(
            output_folder / 'kilosort2_master.m')
        if self.debug:
            print(cmd)
        if "win" in sys.platform:
            cmd_list = ['matlab', '-nosplash', '-nodisplay', '-wait',
                        '-r', 'run {}; quit;'.format(output_folder / 'kilosort2_master.m')]
        else:
            cmd_list = ['matlab', '-nosplash', '-nodisplay',
                        '-r', 'run {}; quit;'.format(output_folder / 'kilosort2_master.m')]

        # retcode = _run_command_and_print_output_split(cmd_list)
        _call_command_split(cmd_list)

    @staticmethod
    def get_result_from_folder(output_folder):
        with (output_folder / 'timestamps.txt').open('r') as f:
            timestamps_strlist = f.readlines()
        timestamps = [float(t) for t in timestamps_strlist]
        with (output_folder / 'labels.txt').open('r') as f:
            labels_strlist = f.readlines()
        labels = [int(l) for l in labels_strlist]
        # we wouldn't need to do the following if we had the recording extractor
        with (output_folder / 'samplefreq.txt').open('r') as f:
            samplefreq_strlist = f.readlines()
        samplefreq = float(samplefreq_strlist[0])

        sorting = se.NumpySortingExtractor()
        sorting.set_sampling_frequency(samplefreq)
        sorting.set_times_labels(timestamps, labels)
        return sorting
