from pathlib import Path
import os
from typing import Union
import sys
import copy
from scipy.io import savemat

import spikeextractors as se
from ..basesorter import BaseSorter
from ..utils.shellscript import ShellScript
from ..sorter_tools import recover_recording

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
    requires_locations = False

    _default_params = {
        'detect_threshold': 5,
        'detect_sign': -1,  # -1 - 1 - 0
        'feature_type': 'wav',
        'scales': 4,
        'min_clus': 20,
        'maxtemp': 0.251,
        'template_sdnum': 3,
        'enable_detect_filter': True,
        'enable_sort_filter': True,
        'detect_filter_fmin': 300,
        'detect_filter_fmax': 3000,
        'detect_filter_order': 4,
        'sort_filter_fmin': 300,
        'sort_filter_fmax': 3000,
        'sort_filter_order': 2,
        'mintemp': 0,
        'max_clus': 200,
        'w_pre': 20,
        'w_post': 44,
        'alignment_window': 10,
        'stdmax': 50,
        'max_spk': 40000,
        'ref_ms': 1.5,
        'interpolation' : True
        }

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
        p = os.getenv('WAVECLUS_PATH', None)
        if p is None:
            return 'unknown'
        else:
            with open(os.path.join(p, 'version.txt'), mode='r', encoding='utf8') as f:
                version = f.readline()
        return version

    @staticmethod
    def set_waveclus_path(waveclus_path: str):
        waveclus_path = str(Path(waveclus_path).absolute())
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
        os.makedirs(str(output_folder), exist_ok=True)
        # Generate mat files in the dataset directory
        for nch in range(recording.get_num_channels()):
            vcFile_mat = str(output_folder / ('raw' +str(nch+1) + '.mat'))
            savemat(vcFile_mat, {'data':recording.get_traces(channel_ids=[nch]), 'sr':recording.get_sampling_frequency()})

    def _run(self, recording, output_folder):
        recording = recover_recording(recording)

        source_dir = Path(__file__).parent
        p = self.params.copy()

        if recording.is_filtered and (p['enable_detect_filter'] or p['enable_sort_filter']):
            print("Warning! The recording is already filtered, but Wave-Clus filters are enabled. You can disable "
                  "filters by setting 'enable_detect_filter' and 'enable_sort_filter' parameters to False")

        if p['detect_sign'] < 0:
            p['detect_sign'] = 'neg'
        elif p['detect_sign'] > 0:
            p['detect_sign'] = 'pos'
        else:
            p['detect_sign'] = 'both'

        if not p['enable_detect_filter']:
            p['detect_filter_order'] = 0
        del p['enable_detect_filter']

        if not p['enable_sort_filter']:
            p['sort_filter_order'] = 0
        del p['enable_sort_filter']

        if p['interpolation']:
            p['interpolation']='y'
        else:
            p['interpolation']='n'

        samplerate = recording.get_sampling_frequency()
        p['sr'] = samplerate

        num_channels = recording.get_num_channels()
        tmpdir = output_folder
        os.makedirs(str(tmpdir), exist_ok=True)

        if self.verbose:
            num_timepoints = recording.get_num_frames()
            duration_minutes = num_timepoints / samplerate / 60
            print('Num. channels = {}, Num. timepoints = {}, duration = {} minutes'.format(
                num_channels, num_timepoints, duration_minutes))

        par_str = ''
        for key, value in p.items():
            if type(value) == str:
                value = '\'{}\''.format(value)
            elif type(value) == bool:
                value = '{}'.format(value).lower()
            par_str += 'par.{} = {};'.format(key, value)

        if self.verbose:
            print('Running waveclus in {tmpdir}...'.format(tmpdir=tmpdir))
        cmd = '''
            addpath(genpath('{waveclus_path}'), '{source_path}');
            {parameters}
            try
                p_waveclus('{tmpdir}', {nChans}, par);
            catch
                fprintf('----------------------------------------');
                fprintf(lasterr());
                quit(1);
            end
            quit(0);
        '''
        cmd = cmd.format(waveclus_path=WaveClusSorter.waveclus_path,source_path=source_dir,
                        tmpdir=tmpdir, nChans=num_channels, parameters=par_str)

        matlab_cmd = ShellScript(cmd, script_path=str(tmpdir / 'run_waveclus.m'), keep_temp_files=True)
        matlab_cmd.write()

        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                cd {tmpdir}
                matlab -nosplash -wait -log -r run_waveclus
            '''.format(tmpdir=tmpdir)
        else:
            shell_cmd = '''
                #!/bin/bash
                cd "{tmpdir}"
                matlab -nosplash -nodisplay -log -r run_waveclus
            '''.format(tmpdir=tmpdir)
        shell_cmd = ShellScript(shell_cmd, script_path=output_folder / f'run_{self.sorter_name}',
                                log_path=output_folder / f'{self.sorter_name}.log', verbose=self.verbose)
        shell_cmd.start()

        retcode = shell_cmd.wait()

        if retcode != 0:
            raise Exception('waveclus returned a non-zero exit code')

        result_fname = str(tmpdir / 'times_results.mat')
        if not os.path.exists(result_fname):
            raise Exception('Result file does not exist: ' + result_fname)


    @staticmethod
    def get_result_from_folder(output_folder):

        output_folder = Path(output_folder)
        result_fname = str(output_folder / 'times_results.mat')
        sorting = se.WaveClusSortingExtractor(file_path=result_fname)
        return sorting
