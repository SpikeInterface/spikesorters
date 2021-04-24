from pathlib import Path
import os
from typing import Union
import sys

import spikeextractors as se
from ..basesorter import BaseSorter
from ..utils.shellscript import ShellScript
from ..sorter_tools import recover_recording

try:
    import h5py

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

PathType = Union[str, Path]


def check_if_installed(combinato_path: Union[str, None]):
    if combinato_path is None:
        return False
    assert isinstance(combinato_path, str)

    if combinato_path.startswith('"'):
        combinato_path = combinato_path[1:-1]
    combinato_path = str(Path(combinato_path).absolute())

    if (Path(combinato_path) / 'css-extract').is_file():
        return True
    else:
        return False


class CombinatoSorter(BaseSorter):
    """
    """

    sorter_name: str = 'combinato'
    combinato_path: Union[str, None] = os.getenv('COMBINATO_PATH', None)
    requires_locations = False
    _default_params = {
        'detect_sign': -1,  # -1 - 1 - 0
        'MaxClustersPerTemp': 5,
        'MinSpikesPerClusterMultiSelect': 15,
        'RecursiveDepth': 1,
        'ReclusterClusters': True,
        'MinInputSizeRecluster': 2000,
        'FirstMatchFactor': .75,
        'SecondMatchFactor': 3,
        'MaxDistMatchGrouping': 1.8,
        'detect_threshold': 5,
        'max_spike_duration': 0.0015,
        'indices_per_spike': 64,
        'index_maximum': 19,
        'upsampling_factor': 3,
        'denoise': True,
        'do_filter': True
    }

    _params_description = {
        'detect_sign': "Use -1 (negative), 1 (positive), or 0 (both) depending "
                       "on the sign of the spikes in the recording",
        'MaxClustersPerTemp': 'How many clusters can be selected at one temperature',
        'MinSpikesPerClusterMultiSelect': 'How many spikes does a cluster need to be selected',
        'RecursiveDepth': 'How many clustering recursions should be run (1 do not recurse)',
        'ReclusterClusters': 'Iteratively recluster big clusters?',
        'MinInputSizeRecluster': 'How many spikes does a cluster need to be re-clustered',
        'FirstMatchFactor': 'How close do spikes have to be in the first template matching step',
        'SecondMatchFactor': 'How close do spikes have to be in the second template matching step',
        'MaxDistMatchGrouping': 'At what cluster distance does grouping stop',
        'detect_threshold': "Threshold for spike detection",
        'max_spike_duration': 'max spike duration in seconds',
        'indices_per_spike': 'samples per spikes',
        'index_maximum': "Number of samples from the beginning of the spike waveform up to (not including) the peak",
        'upsampling_factor': 'upsampling factor',
        'denoise': 'Use denoise filter',
        'do_filter': 'Use bandpass filter'
    }

    sorter_description = """Combinato is a complete data-analysis framework for spike sorting in noisy recordings 
    lasting twelve hours or more. It combines a wavelet-based feature extraction and paramagnetic clustering with 
    multiple stages of template-matching. includes software for artifact rejection, automatic spike sorting, 
    manual optimization, and efficient visualization of results.
    For more information see https://doi:10.1371/journal.pone.0166598"""

    installation_mesg = """\nTo use Combinato run:\n
        >>> git clone https://github.com/jniediek/combinato
    Then inside that folder, run :\n
    >>> python3 setup_options.py
    Finally provide the installation path by setting the COMBINATO_PATH
    environment variables or using CombinatoSorter.set_combinato_path().\n\n

    More information on Combinato at:
        https://github.com/jniediek/combinato/wiki
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @classmethod
    def is_installed(cls):
        return check_if_installed(cls.combinato_path)

    @staticmethod
    def get_sorter_version():
        return 'unknown'

    @staticmethod
    def set_combinato_path(combinato_path: PathType):
        combinato_path = str(Path(combinato_path).absolute())
        CombinatoSorter.combinato_path = combinato_path
        try:
            print("Setting COMBINATO_PATH environment variable for subprocess calls to:", combinato_path)
            os.environ["COMBINATO_PATH"] = combinato_path
        except Exception as e:
            print("Could not set COMBINATO_PATH environment variable:", e)

    def _setup_recording(self, recording, output_folder):
        if not self.is_installed():
            raise Exception(CombinatoSorter.installation_mesg)

        os.makedirs(str(output_folder), exist_ok=True)
        for chid in recording.get_channel_ids():
            # Generate h5 files in the dataset directory
            vcFile_h5 = str(output_folder / ('recording{}.h5'.format(chid)))
            f = h5py.File(vcFile_h5, mode='w')
            f.create_dataset("sr", data=[recording.get_sampling_frequency()], dtype='float32')

            f.create_dataset("data", data=recording.get_traces(channel_ids=[chid]).flatten())
            f.close()

    def _run(self, recording, output_folder):
        recording = recover_recording(recording)
        if len(recording.get_channel_ids()) > 1:
            print("Warning! Combinato is a single-channel sorter, each channel will be sorted independently.")
        p = self.params.copy()
        p['threshold_factor'] = p.pop('detect_threshold')
        sign_thr = p.pop('detect_sign')
        if sign_thr == 0:
            sign_thr = ''
        elif sign_thr == -1:
            sign_thr = '--neg'
        elif sign_thr == 1:
            sign_thr = '--pos'

        if recording.is_filtered and p['do_filter']:
            print("Warning! The recording is already filtered, but Combinato will filter again. You can disable "
                  "filters by setting 'do_filter' to False")

        recording.get_channel_ids()
        tmpdir = output_folder
        os.makedirs(str(tmpdir), exist_ok=True)

        if self.verbose:
            print('Running combinato in {tmpdir}...'.format(tmpdir=tmpdir))
        outFile = open(tmpdir / "local_options.py", "w")
        outFile.writelines("options = {}".format(p))
        outFile.close()

        shell_cmd_ch = '''
            python {css_folder}/css-extract --h5 --files recording{chid}.h5
            python {css_folder}/css-simple-clustering {sign_thr} --datafile recording{chid}/data_recording{chid}.h5'''

        if 'win' in sys.platform and sys.platform != 'darwin':
            extra_cmd = str(tmpdir)[:2]
            shell_cmd_ch = shell_cmd_ch.replace('/', '\\')
        else:
            extra_cmd = '#!/bin/bash'

        shell_cmd = '''
            {extra_cmd}
            cd "{tmpdir}"'''.format(extra_cmd=extra_cmd, tmpdir=tmpdir)

        for chid in recording.get_channel_ids():
            shell_cmd = shell_cmd + shell_cmd_ch.format(extra_cmd=extra_cmd, tmpdir=tmpdir,
                                                        css_folder=CombinatoSorter.combinato_path,
                                                        sign_thr=sign_thr, chid=chid)
        shell_cmd = ShellScript(shell_cmd, script_path=output_folder / f'run_{self.sorter_name}',
                                log_path=output_folder / f'{self.sorter_name}.log', verbose=self.verbose)
        shell_cmd.start()

        retcode = shell_cmd.wait()

        if retcode != 0:
            raise Exception('combinato returned a non-zero exit code')

    @staticmethod
    def get_result_from_folder(output_folder):

        output_folder = Path(output_folder)
        sortings = []
        recording_folders = [f for f in os.listdir(output_folder) if (os.path.isdir(os.path.join(output_folder, f)) and f.startswith('recording'))]

        for rf in recording_folders:
            result_fname = str(output_folder / rf)
            sorting = se.CombinatoSortingExtractor(datapath=result_fname)
            for u in sorting.get_unit_ids():
                sorting.set_unit_property(unit_id=u, property_name='channel_id', value=int(rf[9:]))
            sortings.append(sorting)

        return se.MultiSortingExtractor(sortings)
