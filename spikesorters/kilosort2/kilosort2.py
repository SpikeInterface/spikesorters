from pathlib import Path
import os
import sys
import numpy as np
from typing import Union
import shutil
import json

import spikeextractors as se
from ..basesorter import BaseSorter
from ..utils.shellscript import ShellScript
from ..sorter_tools import get_git_commit


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


class Kilosort2Sorter(BaseSorter):
    """
    """

    sorter_name: str = 'kilosort2'
    kilosort2_path: Union[str, None] = os.getenv('KILOSORT2_PATH', None)
    installed = check_if_installed(kilosort2_path)
    requires_locations = False

    _default_params = {
        'detect_threshold': 5,
        'projection_threshold': [10, 4],
        'preclust_threshold': 8,
        'car': True,
        'minFR': 0.1,
        'minfr_goodchannels': 0.1,
        'freq_min': 150,
        'sigmaMask': 30,
        'nPCs': 3,
        'ntbuff': 64,
        'nfilt_factor': 4,
        'NT': None,
        'keep_good_only': False
    }

    installation_mesg = """\nTo use Kilosort2 run:\n
        >>> git clone https://github.com/MouseLand/Kilosort2
    and provide the installation path by setting the KILOSORT2_PATH
    environment variables or using Kilosort2Sorter.set_kilosort2_path().\n\n

    More information on Kilosort2 at:
        https://github.com/MouseLand/Kilosort2
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @staticmethod
    def get_sorter_version():
        commit = get_git_commit(os.getenv('KILOSORT2_PATH', None))
        if commit is None:
            return 'unknown'
        else:
            return 'git-' + commit

    @staticmethod
    def set_kilosort2_path(kilosort2_path: str):
        Kilosort2Sorter.kilosort2_path = kilosort2_path
        Kilosort2Sorter.installed = check_if_installed(Kilosort2Sorter.kilosort2_path)
        try:
            print("Setting KILOSORT2_PATH environment variable for subprocess calls to:", kilosort2_path)
            os.environ["KILOSORT2_PATH"] = kilosort2_path
        except Exception as e:
            print("Could not set KILOSORT2_PATH environment variable:", e)

    def _setup_recording(self, recording, output_folder):
        source_dir = Path(Path(__file__).parent)
        p = self.params

        if not check_if_installed(Kilosort2Sorter.kilosort2_path):
            raise Exception(Kilosort2Sorter.installation_mesg)
        assert isinstance(Kilosort2Sorter.kilosort2_path, str)

        # prepare electrode positions for this group (only one group, the split is done in basesorter)
        groups = [1] * recording.get_num_channels()
        positions = np.array(recording.get_channel_locations())
        if positions.shape[1] != 2:
            raise RuntimeError("3D 'location' are not supported. Set 2D locations instead")

        # save binary file
        input_file_path = output_folder / 'recording'
        recording.write_to_binary_dat_format(input_file_path, dtype='int16', chunk_mb=500)

        if p['car']:
            use_car = 1
        else:
            use_car = 0

        # read the template txt files
        with (source_dir / 'kilosort2_master.m').open('r') as f:
            kilosort2_master_txt = f.read()
        with (source_dir / 'kilosort2_config.m').open('r') as f:
            kilosort2_config_txt = f.read()
        with (source_dir / 'kilosort2_channelmap.m').open('r') as f:
            kilosort2_channelmap_txt = f.read()

        # make substitutions in txt files
        kilosort2_master_txt = kilosort2_master_txt.format(
            kilosort2_path=str(
                Path(Kilosort2Sorter.kilosort2_path).absolute()),
            output_folder=str(output_folder),
            channel_path=str(
                (output_folder / 'kilosort2_channelmap.m').absolute()),
            config_path=str((output_folder / 'kilosort2_config.m').absolute()),
        )

        if p['NT'] is None:
            p['NT'] = 64 * 1024 + p['ntbuff']
        else:
            p['NT'] = p['NT'] // 32 * 32  # make sure is multiple of 32

        kilosort2_config_txt = kilosort2_config_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            dat_file=str((output_folder / 'recording.dat').absolute()),
            projection_threshold=p['projection_threshold'],
            preclust_threshold=p['preclust_threshold'],
            minfr_goodchannels=p['minfr_goodchannels'],
            minFR=p['minFR'],
            freq_min=p['freq_min'],
            sigmaMask=p['sigmaMask'],
            kilo_thresh=p['detect_threshold'],
            use_car=use_car,
            nPCs=int(p['nPCs']),
            ntbuff=int(p['ntbuff']),
            nfilt_factor=int(p['nfilt_factor']),
            NT=int(p['NT'])
        )

        kilosort2_channelmap_txt = kilosort2_channelmap_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            xcoords=[p[0] for p in positions],
            ycoords=[p[1] for p in positions],
            kcoords=groups
        )

        for fname, txt in zip(['kilosort2_master.m', 'kilosort2_config.m',
                               'kilosort2_channelmap.m'],
                              [kilosort2_master_txt, kilosort2_config_txt,
                               kilosort2_channelmap_txt]):
            with (output_folder / fname).open('w') as f:
                f.write(txt)

        shutil.copy(str(source_dir.parent / 'utils' / 'writeNPY.m'), str(output_folder))
        shutil.copy(str(source_dir.parent / 'utils' / 'constructNPYheader.m'), str(output_folder))

    def _run(self, recording, output_folder):
        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        cd {tmpdir}
                        matlab -nosplash -nodisplay -wait -r kilosort2_master
                    '''.format(tmpdir=output_folder)
        else:
            shell_cmd = '''
                        #!/bin/bash
                        cd "{tmpdir}"
                        matlab -nosplash -nodisplay -r kilosort2_master
                    '''.format(tmpdir=output_folder)
        shell_script = ShellScript(shell_cmd, script_path=output_folder / f'run_{self.sorter_name}')
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('kilosort2 returned a non-zero exit code')

    @staticmethod
    def get_result_from_folder(output_folder):
        output_folder = Path(output_folder)
        with (output_folder / 'spikeinterface_params.json').open('r') as f:
            sorter_params = json.load(f)['sorter_params']
        sorting = se.KiloSortSortingExtractor(folder_path=output_folder, keep_good_only=sorter_params['keep_good_only'])
        return sorting
