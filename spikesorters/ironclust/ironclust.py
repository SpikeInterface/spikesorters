import os
from pathlib import Path
from typing import Union

import spikeextractors as se

from .mdarecordingextractor2 import MdaRecordingExtractor2
from .shellscript import ShellScript
from ..basesorter import BaseSorter


def check_if_installed(ironclust_path: Union[str, None]):
    if ironclust_path is None:
        return False
    assert isinstance(ironclust_path, str)

    if ironclust_path.startswith('"'):
        ironclust_path = ironclust_path[1:-1]
    ironclust_path = str(Path(ironclust_path).absolute())

    if (Path(ironclust_path) / 'matlab' / 'irc.m').is_file():
        return True
    else:
        return False


class IronClustSorter(BaseSorter):
    """
    """

    sorter_name: str = 'ironclust'
    ironclust_path: Union[str, None] = os.getenv('IRONCLUST_PATH', None)
    installed = check_if_installed(ironclust_path)

    _default_params = dict(
        detect_sign=-1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        adjacency_radius=50,  # Use -1 to include all channels in every neighborhood
        adjacency_radius_out=75,  # Use -1 to include all channels in every neighborhood
        detect_threshold=4.5,  # detection threshold
        prm_template_name='',  # .prm template file name
        freq_min=300,
        freq_max=6000,
        merge_thresh=0.985,  # Threshold for automated merging
        pc_per_chan=2,  # Number of principal components per channel
        whiten=True,  # Whether to do channel whitening as part of preprocessing
        filter_type='bandpass',  # none, bandpass, wiener, fftdiff, ndiff
        filter_detect_type='none',  # none, bandpass, wiener, fftdiff, ndiff
        common_ref_type='none',  # none, mean, median
        batch_sec_drift=300,  # batch duration in seconds. clustering time duration
        step_sec_drift=20,  # compute anatomical similarity every n sec
        knn=30,  # K nearest neighbors
        min_count=30,  # Minimum cluster size
        fGpu=True,  # Use GPU if available
        fft_thresh=8,  # FFT-based noise peak threshold
        fft_thresh_low=0,  # FFT-based noise peak lower threshold (set to 0 to disable dual thresholding scheme
        nSites_whiten=32,  # Number of adjacent channels to whiten
        feature_type='gpca',  # gpca, pca, vpp, vmin, vminmax, cov, energy, xcov
        delta_cut=1,  # Cluster detection threshold (delta-cutoff)
        post_merge_mode=1,  # post merge mode
        sort_mode=1  # sort mode
    )

    installation_mesg = """\nTo use IronClust run:\n
        >>> git clone https://github.com/jamesjun/ironclust
    and provide the installation path by setting the IRONCLUST_PATH
    environment variables or using IronClustSorter.set_ironclust_path().\n\n
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @staticmethod
    def set_ironclust_path(ironclust_path: str):
        IronClustSorter.ironclust_path = ironclust_path
        IronClustSorter.installed = check_if_installed(IronClustSorter.ironclust_path)
        try:
            print("Setting IRONCLUST_PATH environment variable for subprocess calls to:", ironclust_path)
            os.environ["IRONCLUST_PATH"] = ironclust_path
        except Exception as e:
            print("Could not set IRONCLUST_PATH environment variable:", e)

    def _setup_recording(self, recording: se.RecordingExtractor, output_folder: Path):
        if not check_if_installed(IronClustSorter.ironclust_path):
            raise Exception(IronClustSorter.installation_mesg)
        assert isinstance(IronClustSorter.ironclust_path, str)

        dataset_dir = output_folder / 'ironclust_dataset'
        # Generate three files in the dataset directory: raw.mda, geom.csv, params.json
        MdaRecordingExtractor2.write_recording(recording=recording, save_path=str(dataset_dir), _preserve_dtype=True)

    def _run(self, recording: se.RecordingExtractor, output_folder: Path):
        dataset_dir = output_folder / 'ironclust_dataset'
        source_dir = Path(__file__).parent

        samplerate = recording.get_sampling_frequency()

        num_channels = recording.get_num_channels()
        num_timepoints = recording.get_num_frames()
        duration_minutes = num_timepoints / samplerate / 60
        if self.debug:
            print('Num. channels = {}, Num. timepoints = {}, duration = {} minutes'.format(
            num_channels, num_timepoints, duration_minutes))

        if self.debug:
            print('Creating argfile.txt...')
        txt = ''
        for key0, val0 in self.params.items():
            txt += '{}={}\n'.format(key0, val0)
        txt += 'samplerate={}\n'.format(samplerate)
        with (dataset_dir / 'argfile.txt').open('w') as f:
            f.write(txt)

        tmpdir = output_folder / 'tmp'
        os.makedirs(str(tmpdir), exist_ok=True)
        if self.debug:
            print('Running ironclust in {tmpdir}...'.format(tmpdir=str(tmpdir)))
        cmd = '''
            addpath('{source_dir}');
            addpath('{ironclust_path}', '{ironclust_path}/matlab', '{ironclust_path}/matlab/mdaio');
            try
                p_ironclust('{tmpdir}', '{dataset_dir}/raw.mda', '{dataset_dir}/geom.csv', '', '', '{tmpdir}/firings.mda', '{dataset_dir}/argfile.txt');
            catch
                fprintf('----------------------------------------');
                fprintf(lasterr());
                quit(1);
            end
            quit(0);
        '''
        cmd = cmd.format(ironclust_path=IronClustSorter.ironclust_path, tmpdir=str(tmpdir),
                         dataset_dir=str(dataset_dir), source_dir=str(source_dir))

        matlab_cmd = ShellScript(cmd, script_path=str(tmpdir / 'run_ironclust.m'))
        matlab_cmd.write()

        shell_cmd = '''
            #!/bin/bash
            cd {tmpdir}
            matlab -nosplash -nodisplay -r run_ironclust
        '''.format(tmpdir=str(tmpdir))
        shell_script = ShellScript(shell_cmd, script_path=str(tmpdir / 'run_ironclust.sh'))
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('ironclust returned a non-zero exit code')

        result_fname = str(tmpdir / 'firings.mda')
        if not os.path.exists(result_fname):
            raise Exception('Result file does not exist: ' + result_fname)

        samplerate_fname = str(tmpdir / 'samplerate.txt')
        with open(samplerate_fname, 'w') as f:
            f.write('{}'.format(samplerate))

    @staticmethod
    def get_result_from_folder(output_folder: Union[str, Path]):
        output_folder = Path(output_folder)
        tmpdir = output_folder / 'tmp'

        result_fname = str(tmpdir / 'firings.mda')
        samplerate_fname = str(tmpdir / 'samplerate.txt')
        with open(samplerate_fname, 'r') as f:
            samplerate = float(f.read())

        sorting = se.MdaSortingExtractor(firings_file=result_fname, sampling_frequency=samplerate)

        return sorting
