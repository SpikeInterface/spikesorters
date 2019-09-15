import mlprocessors as mlpr
import os
import time
import numpy as np
from os.path import join
from subprocess import Popen, PIPE
import shlex
import random
import string
import shutil
from spikeforest import mdaio
import spikeextractors as se
from spikeforest import SFMdaRecordingExtractor, SFMdaSortingExtractor
from mountaintools import client as mt
import traceback
import json
from .install_waveclus import install_waveclus


class Waveclus(mlpr.Processor):
    """
    Wave_clus wrapper
      written by J. James Jun, May 21, 2019

    [Optional: Installation instruction in SpikeForest environment]
    1. Run `git clone https://github.com/csn-le/wave_clus.git`
    2. Activate conda environment for SpikeForest
    3. Create `WAVECLUS_PATH_DEV`

    Algorithm website:
    https://github.com/csn-le/wave_clus/wiki
    """

    NAME = 'waveclus'
    VERSION = '0.0.3'
    ENVIRONMENT_VARIABLES = [
        'NUM_WORKERS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS', 'TEMPDIR']
    ADDITIONAL_FILES = ['*.m', '*.prm']
    CONTAINER = None
    LOCAL_MODULES = ['../../spikeforest']

    recording_dir = mlpr.Input('Directory of recording', directory=True)
    firings_out = mlpr.Output('Output firings file')

    def run(self):
        tmpdir = _get_tmpdir('waveclus')

        try:
            recording = SFMdaRecordingExtractor(self.recording_dir)
            params = read_dataset_params(self.recording_dir)
            # if len(self.channels) > 0:
            #     recording = se.SubRecordingExtractor(
            #         parent_recording=recording, channel_ids=self.channels)
            if not os.path.exists(tmpdir):
                os.mkdir(tmpdir)

            all_params = dict()
            for param0 in self.PARAMETERS:
                all_params[param0.name] = getattr(self, param0.name)
            sorting = waveclus_helper(
                recording=recording,
                tmpdir=tmpdir,
                params=params,
                **all_params,
            )
            SFMdaSortingExtractor.write_sorting(
                sorting=sorting, save_path=self.firings_out)
        except:
            if os.path.exists(tmpdir):
                if not getattr(self, '_keep_temp_files', False):
                    print('erased temp file 1')
                    shutil.rmtree(tmpdir)
            raise
        if not getattr(self, '_keep_temp_files', False):
            print('erased temp file 2')
            shutil.rmtree(tmpdir)


def waveclus_helper(
        *,
        recording,  # Recording object
        tmpdir,  # Temporary working directory
        params=dict(),
        **kwargs):

    waveclus_path = os.environ.get('WAVECLUS_PATH_DEV', None)
    if waveclus_path:
        print('Using waveclus from WAVECLUS_PATH_DEV directory: {}'.format(waveclus_path))
    else:
        try:
            print('Auto-installing waveclus.')
            waveclus_path = install_waveclus(
                repo='https://github.com/csn-le/wave_clus.git',
                commit='248d15c7eaa2b45b15e4488dfb9b09bfe39f5341'
            )
        except:
            traceback.print_exc()
            raise Exception('Problem installing waveclus. You can set the WAVECLUS_PATH_DEV to force to use a particular path.')
    print('Using waveclus from: {}'.format(waveclus_path))

    # source_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_dir = os.path.join(tmpdir, 'waveclus_dataset')
    # Generate three files in the dataset directory: raw.mda, geom.csv, params.json
    SFMdaRecordingExtractor.write_recording(
        recording=recording, save_path=dataset_dir, params=params, _preserve_dtype=True)

    samplerate = recording.get_sampling_frequency()

    print('Reading timeseries header...')
    raw_mda = os.path.join(dataset_dir, 'raw.mda')
    HH = mdaio.readmda_header(raw_mda)
    num_channels = HH.dims[0]
    num_timepoints = HH.dims[1]
    duration_minutes = num_timepoints / samplerate / 60
    print('Num. channels = {}, Num. timepoints = {}, duration = {} minutes'.format(
        num_channels, num_timepoints, duration_minutes))

    # new method
    source_path = os.path.dirname(os.path.realpath(__file__))
    print('Running waveclus in {tmpdir}...'.format(tmpdir=tmpdir))
    cmd = '''
        addpath(genpath('{waveclus_path}'), '{source_path}', '{source_path}/mdaio');
        try
            p_waveclus('{tmpdir}', '{dataset_dir}/raw.mda', '{tmpdir}/firings.mda', {samplerate});
        catch
            fprintf('----------------------------------------');
            fprintf(lasterr());
            quit(1);
        end
        quit(0);
    '''
    cmd = cmd.format(waveclus_path=waveclus_path, tmpdir=tmpdir, dataset_dir=dataset_dir, source_path=source_path, samplerate=samplerate)

    matlab_cmd = mlpr.ShellScript(cmd, script_path=tmpdir + '/run_waveclus.m', keep_temp_files=True)
    matlab_cmd.write()

    shell_cmd = '''
        #!/bin/bash
        cd {tmpdir}
        matlab -nosplash -nodisplay -r run_waveclus
    '''.format(tmpdir=tmpdir)
    shell_cmd = mlpr.ShellScript(shell_cmd, script_path=tmpdir + '/run_waveclus.sh', keep_temp_files=True)
    shell_cmd.write(tmpdir + '/run_waveclus.sh')
    shell_cmd.start()

    retcode = shell_cmd.wait()

    if retcode != 0:
        raise Exception('waveclus returned a non-zero exit code')

    # parse output
    result_fname = tmpdir + '/firings.mda'
    if not os.path.exists(result_fname):
        raise Exception('Result file does not exist: ' + result_fname)

    firings = mdaio.readmda(result_fname)
    sorting = se.NumpySortingExtractor()
    sorting.set_times_labels(firings[1, :], firings[2, :])
    return sorting


def _read_text_file(fname):
    with open(fname) as f:
        return f.read()


def _write_text_file(fname, str):
    with open(fname, 'w') as f:
        f.write(str)


def _run_command_and_print_output(command):
    with Popen(shlex.split(command), stdout=PIPE, stderr=PIPE) as process:
        while True:
            output_stdout = process.stdout.readline()
            output_stderr = process.stderr.readline()
            if (not output_stdout) and (not output_stderr) and (process.poll() is not None):
                break
            if output_stdout:
                print(output_stdout.decode())
            if output_stderr:
                print(output_stderr.decode())
        rc = process.poll()
        return rc


def read_dataset_params(dsdir):
    # ca = _load_required_modules()
    fname1 = dsdir + '/params.json'
    fname2 = mt.realizeFile(path=fname1)
    if not fname2:
        raise Exception('Unable to find file: ' + fname1)
    if not os.path.exists(fname2):
        raise Exception('Dataset parameter file does not exist: ' + fname2)
    with open(fname2) as f:
        return json.load(f)


# To be shared across sorters (2019.05.05)
def _get_tmpdir(sorter_name):
    code = ''.join(random.choice(string.ascii_uppercase) for x in range(10))
    tmpdir0 = os.environ.get('TEMPDIR', '/tmp')
    tmpdir = os.path.join(tmpdir0, '{}-tmp-{}'.format(sorter_name, code))
    # reset the output folder
    if os.path.exists(tmpdir):
        shutil.rmtree(str(tmpdir))
    else:
        os.makedirs(tmpdir)
    return tmpdir
