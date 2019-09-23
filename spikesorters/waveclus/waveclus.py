from pathlib import Path
import os
from typing import Union
import sys
import copy

import spikeextractors as se
from ..basesorter import BaseSorter
from ..utils.ssmdarecordingextractor import SSMdaRecordingExtractor
from ..utils.shellscript import ShellScript


def check_if_installed(waveclus_path: Union[str, None]):
    if waveclus_path is None:
        return False
    assert isinstance(waveclus_path, str)

    if waveclus_path.startswith('"'):
        waveclus_path = waveclus_path[1:-1]
    waveclus_path = str(Path(waveclus_path).absolute())

    if (Path(waveclus_path) / 'wave_clus.m').is_file():
        return True
    else:
        return False


class WaveClusSorter(BaseSorter):
    """
    """

    sorter_name: str = 'waveclus'
    waveclus_path: Union[str, None] = os.getenv('WAVECLUS_PATH', None)
    installed = check_if_installed(waveclus_path)

    _default_params = {
        'detect_threshold': 5,
        'detect_sign': -1,  # -1 - 1 - 0
        'feature_type': 'wav',
        'scales': 4,
        'min_clus': 20,
        'maxtemp': 0.251,
        'template_sdnum': 3,
    }

    _extra_gui_params = [
        {'name': 'detect_threshold', 'type': 'float', 'value': 5.0, 'default': 5.0,
         'title': "Relative detection threshold"},
        {'name': 'detect_sign', 'type': 'int', 'value': -1, 'default': -1,
         'title': "Use -1, 0, or 1, depending on the sign of the spikes in the recording"},
        {'name': 'feature_type', 'type': 'str', 'value': 'wav', 'default': 'wav',
         'title': "Feature type ('wav', 'pca')"},
        {'name': 'scales', 'type': 'int', 'value': 4, 'default': 4, 'title': "Number of wavelet scales"},
        {'name': 'min_clus', 'type': 'int', 'value': 20, 'default': 20, 'title': "Minimum size of a cluster"},
        {'name': 'maxtemp', 'type': 'float', 'value': 0.251, 'default': 0.251, 'title': "Maximum temperature for SPC"},
        {'name': 'template_sdnum', 'type': 'int', 'value': 3, 'default': 3,
         'title': "Max radius of cluster in std devs"},
    ]

    _gui_params = copy.deepcopy(BaseSorter.sorter_gui_params)
    for param in _extra_gui_params:
        _gui_params.append(param)

    installation_mesg = """\nTo use WaveClus run:\n
        >>> git clone https://github.com/csn-le/wave_clus
    and provide the installation path by setting the WAVECLUS_PATH
    environment variables or using WaveClusSorter.set_waveclus_path().\n\n

    More information on WaveClus at:
        https://github.com/csn-le/wave_clus/wiki
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @staticmethod
    def get_sorter_version():
        return 'unknown'

    @staticmethod
    def set_waveclus_path(waveclus_path: str):
        WaveClusSorter.waveclus_path = waveclus_path
        WaveClusSorter.installed = check_if_installed(WaveClusSorter.waveclus_path)
        try:
            print("Setting WAVECLUS_PATH environment variable for subprocess calls to:", waveclus_path)
            os.environ["WAVECLUS_PATH"] = waveclus_path
        except Exception as e:
            print("Could not set WAVECLUS_PATH environment variable:", e)

    def _setup_recording(self, recording, output_folder):
        if not check_if_installed(WaveClusSorter.waveclus_path):
            raise Exception(WaveClusSorter.installation_mesg)
        assert isinstance(WaveClusSorter.waveclus_path, str)

        dataset_dir = output_folder / 'waveclus_dataset'
        # Generate three files in the dataset directory: raw.mda, geom.csv, params.json
        SSMdaRecordingExtractor.write_recording(recording=recording, save_path=str(dataset_dir), _preserve_dtype=True)

    def _run(self, recording, output_folder):
        dataset_dir = output_folder / 'waveclus_dataset'
        source_dir = Path(__file__).parent
        p = self.params

        if p['detect_sign'] < 0:
            detect_sign = 'neg'
        elif p['detect_sign'] > 0:
            detect_sign = 'pos'
        else:
            detect_sign = 'both'

        samplerate = recording.get_sampling_frequency()

        num_channels = recording.get_num_channels()
        num_timepoints = recording.get_num_frames()
        duration_minutes = num_timepoints / samplerate / 60

        tmpdir = output_folder / 'tmp'
        os.makedirs(str(tmpdir), exist_ok=True)

        if self.verbose:
            print('Num. channels = {}, Num. timepoints = {}, duration = {} minutes'.format(
                num_channels, num_timepoints, duration_minutes))

        # new method
        utils_path = source_dir.parent / 'utils'
        if self.verbose:
            print('Running waveclus in {tmpdir}...'.format(tmpdir=tmpdir))
        cmd = '''
            addpath(genpath('{waveclus_path}'), '{source_path}', '{utils_path}/mdaio');
            try
                p_waveclus('{tmpdir}', '{dataset_dir}/raw.mda', '{tmpdir}/firings.mda', {samplerate}, ...
                '{detect_sign}', '{feature_type}', {scales}, {detect_threshold}, {min_clus}, {maxtemp}, ...
                 {template_sdnum});
            catch
                fprintf('----------------------------------------');
                fprintf(lasterr());
                quit(1);
            end
            quit(0);
        '''
        cmd = cmd.format(waveclus_path=WaveClusSorter.waveclus_path, utils_path=utils_path, dataset_dir=dataset_dir,
                         source_path=source_dir, samplerate=samplerate, detect_sign=detect_sign, tmpdir=tmpdir,
                         feature_type=p['feature_type'], scales=p['scales'], detect_threshold=p['detect_threshold'],
                         min_clus=p['min_clus'], maxtemp=p['maxtemp'], template_sdnum=p['template_sdnum'])

        matlab_cmd = ShellScript(cmd, script_path=str(tmpdir / 'run_waveclus.m'), keep_temp_files=True)
        matlab_cmd.write()

        if "win" in sys.platform:
            shell_cmd = '''
                #!/bin/bash
                cd {tmpdir}
                matlab -nosplash -nodisplay -wait -r run_waveclus
            '''.format(tmpdir=tmpdir)
        else:
            shell_cmd = '''
                #!/bin/bash
                cd {tmpdir}
                matlab -nosplash -nodisplay -r run_waveclus
            '''.format(tmpdir=tmpdir)
        shell_cmd = ShellScript(shell_cmd, script_path=str(tmpdir / 'run_waveclus.sh'), keep_temp_files=True)
        shell_cmd.write(str(tmpdir / 'run_waveclus.sh'))
        shell_cmd.start()

        retcode = shell_cmd.wait()

        if retcode != 0:
            raise Exception('waveclus returned a non-zero exit code')

        result_fname = str(tmpdir / 'firings.mda')
        if not os.path.exists(result_fname):
            raise Exception('Result file does not exist: ' + result_fname)

        samplerate_fname = str(tmpdir / 'samplerate.txt')
        with open(samplerate_fname, 'w') as f:
            f.write('{}'.format(samplerate))

    @staticmethod
    def get_result_from_folder(output_folder):
        output_folder = Path(output_folder)
        tmpdir = output_folder / 'tmp'

        result_fname = str(tmpdir / 'firings.mda')
        samplerate_fname = str(tmpdir / 'samplerate.txt')
        with open(samplerate_fname, 'r') as f:
            samplerate = float(f.read())

        sorting = se.MdaSortingExtractor(file_path=result_fname, sampling_frequency=samplerate)

        return sorting
