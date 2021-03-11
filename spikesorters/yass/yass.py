import copy
from pathlib import Path
import os
import numpy as np
from numpy.lib.format import open_memmap
import sys

import spikeextractors as se
from ..basesorter import BaseSorter
from ..utils.shellscript import ShellScript
from ..sorter_tools import recover_recording

try:
    import yaml
    import yass

    HAVE_YASS = True
except ImportError:
    HAVE_YASS = False


class YassSorter(BaseSorter):
    """
    """

    sorter_name = 'yass'
    requires_locations = False

    # #################################################

    _default_params = {
        'dtype': 'int16',  # the only datatype that Yass currently accepts;

        # Filtering and processing params
        'freq_min': 300,  # "High-pass filter cutoff frequency",
        'freq_max': 0.3,  # "Low-pass filter cutoff frequency as proportion of sampling rate",
        'neural_nets_path': None,  # default NNs are set to None - Yass will always retrain on dataset;
        'multi_processing': 1,  # 0: single core; 1: multi CPU core
        'n_processors': 1,  # default is a single core; autosearch for more cores
        'n_gpu_processors': 1,  # default is the first installed GPU
        'n_sec_chunk': 10,  # Length of processing chunk in seconds for multi-processing stages
        'n_sec_chunk_gpu_detect': 0.5,  # n_sec_chunk for gpu detection (lower if you get memory error during detection)
        'n_sec_chunk_gpu_deconv': 5,  # n_sec_chunk for gpu deconvolution (lower if you get memory error during deconv)
        'gpu_id': 0,  # which gpu to use, default is 0, i.e. first gpu;
        'generate_phy': 0,  # generate phy visualization files; 0 - do not run; 1: generate phy files
        'phy_percent_spikes': 0.05,
        # generate phy visualization files; ratio of spikes that are processed for phy visualization
        # decrease if memory issues are present

        # params related to NN and clustering;
        'spatial_radius': 70,  # channels spatial radius to consider them neighbors, see
        # yass.geometry.find_channel_neighbors for details

        'spike_size_ms': 5,  # temporal length of templates in ms. It must capture
        # the full shape of waveforms on all channels
        # (reminder: there is a propagation delay in waveform shape across channels)
        # but longer means slower
        'clustering_chunk': [0, 300],  # time (in sec) to run clustering and get initial templates
        # leave blank to run clustering step on entire recording;
        # deconv is then run on the entire dataset using clustering stage templates

        # Params for deconv stage
        'update_templates': 0,  # update templates during deconvolution step
        'neuron_discover': 0,  # recluster during deconvlution and search for new stable neurons;
        'template_update_time': 300,  # if tempaltes being udpated, time (in sec) of segment in which to search for
        # new clusters

        # Defatul params for converting raw data to required formats.
        'chunk_mb': 500,  # chunk of data
        'n_jobs_bin': 1  # number of cores?
    }

    _params_description = {

        'dtype': 'int16 : the only datatype that Yass currently accepts',

        # Filtering and processing params
        'freq_min': "300; High-pass filter cutoff frequency",
        'freq_max': "0.3; Low-pass filter cutoff frequency as proportion of sampling rate",
        'neural_nets_path': ' None;  default NNs are set to None - Yass will always retrain on dataset',
        'multi_processing': '1; 0: single core; 1: multi CPU core',
        'n_processors': ' 1; default is a single core; TODO: auto-detect # of corse on node',
        'n_gpu_processors': '1: default is the first installed GPU',
        'n_sec_chunk': '10;  Length of processing chunk in seconds for multi-processing stages. Lower this if running out of memory',
        'n_sec_chunk_gpu_detect': '0.5; n_sec_chunk for gpu detection (lower if you get memory error during detection)',
        'n_sec_chunk_gpu_deconv': '5; n_sec_chunk for gpu deconvolution (lower if you get memory error during deconv)',
        'gpu_id': '0; which gpu ID to use, default is 0, i.e. first gpu',
        'generate_phy': '1; generate phy visualization files; 0 - do not run; 1: generate phy files',
        'phy_percent_spikes': '0.05;  ratio of spikes that are processed for phy visualization; decrease if memory issues are present',

        # params related to NN and clustering;
        'spatial_radius': '70; spatial radius to consider 2 channels neighbors; required for NN stages to work',
        'spike_size_ms': '5; temporal length of templates in ms; longer is more processing time, but slight more accurate',
        # but longer means slower
        'clustering_chunk': '[0, 300]; period of time (in sec) to run clustering and get initial templates; leave blank to run clustering step on entire recording;',

        # Params for deconv stage
        'update_templates': '0; update templates during deconvolution step 1; do not update 0',
        'neuron_discover': '0, recluster during deconvlution and search for new stable neurons: 1; do not recluser 0',
        'template_update_time': '300; if reculstiner on, time (in sec) of segment in which to search for new clusters ',

        # Defatul params for converting raw data to required formats.
        'chunk_mb': '500; chunk of data to be processed by a single core',
        'n_jobs_bin': '1; number of cores to do data conversion',

    }

    # #################################################

    sorter_description = """Yass is a deconvolution and neural network based spike sorting algorithm designed for
                            recordings with no drift (such as retinal recordings).

                            For more information see https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1"""

    installation_mesg = """\nTo install Yass run:\n
                            pip install yass-algorithm

                            Yass can be run in 2 modes:

                            1.  Retraining Neural Networks (Default)

                            import spikesorters as ss
                            import spikeextractors as se
                            rec, sort = se.example_datasets.toy_example(duration=300)
                            sorting_yass = ss.run_yass(rec, '/home/cat/Downloads/test2')


                            2.  Using previously trained Neural Networks:
                            ...
                            sorting_yass = ss.run_yass(rec, '/home/cat/Downloads/test2', neural_nets_path=PATH_TO_NEURAL_NETS)

                            For any installation or operation issues please visit: https://github.com/paninski-lab/yass

                        """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

        source_dir = Path(__file__).parent
        config_default_location = os.path.join(source_dir, 'config_default.yaml')

        with open(config_default_location) as file:
            self.yass_params = yaml.load(file, Loader=yaml.FullLoader)

        self.neural_nets_path = None

    @classmethod
    def is_installed(cls):
        return HAVE_YASS

    @staticmethod
    def get_sorter_version():
        return yass.__version__

    def _setup_recording(self, recording, output_folder):
        p = self.params

        #################################################################
        #################### UPDATE ROOT FOLDER #########################
        #################################################################
        self.yass_params['data']['root_folder'] = str(output_folder)

        #################################################################
        #################### GEOMETRY FILE GENERATION ###################
        #################################################################
        probe_file_txt = os.path.join(output_folder, 'geom.txt')
        geom_txt = recording.get_channel_locations()
        np.savetxt(probe_file_txt, geom_txt)

        #################################################################
        #################### UPDATE SAMPLING RATE #######################
        #################################################################
        self.yass_params['recordings']['sampling_rate'] = recording.get_sampling_frequency()

        #################################################################
        #################### UPDATE N_CHAN  #############################
        #################################################################
        self.yass_params['recordings']['n_channels'] = recording.get_num_channels()

        #################################################################
        #################### SAVE RAW INT16 data ########################
        #################################################################
        input_file_path = os.path.join(output_folder, 'data.bin')
        recording.write_to_binary_dat_format(input_file_path,
                                             dtype='int16',  # HARD CODE THIS FOR YASS
                                             chunk_mb=p["chunk_mb"],
                                             n_jobs=p["n_jobs_bin"],
                                             verbose=self.verbose)

        retrain = False
        if self.params['neural_nets_path'] is None:
            self.params['neural_nets_path'] = os.path.join(output_folder,
                                                           'tmp',
                                                           'nn_train')
            retrain = True

        #################################################################
        ######## MERGE Yass config parameters with self.params ##########
        #################################################################
        # MERGE yass_params with self.params that could be changed by the user
        self.merge_params_dict()

        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                    'config.yaml')
        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.merge_params, file)

        #################################################################
        ############ RUN NN TRAINING ON EXISTING DATASET ################
        #################################################################
        self.neural_nets_path = p['neural_nets_path']

        if retrain:
            # retrain NNs
            self.train(recording, output_folder)

            # update NN folder location
            neural_nets_path = os.path.join(output_folder,
                                            'tmp',
                                            'nn_train')

        #################################################################
        ####################### OR LOAD PREVIOUS NNS ####################
        #################################################################
        else:
            print("USING PREVIOUSLY TRAINED NNs FROM THIS LOCATION: ",
                  self.neural_nets_path)
            # use previuosly trained NN folder location
            neural_nets_path = self.neural_nets_path

        self.neural_nets_update_location(output_folder, neural_nets_path)

    def merge_params_dict(self):
        ''' This function merges self.params with self.yass_params to
            make a larger exposed params dictionary
        '''
        # self.params
        # self.yass_params

        self.merge_params = self.yass_params

        self.merge_params['preprocess']['filter']['low_pass_freq'] = self.params['freq_min']
        self.merge_params['preprocess']['filter']['high_factor'] = self.params['freq_max']

        self.merge_params['neuralnetwork']['detect']['filename'] = os.path.join(
            self.params['neural_nets_path'],
            'detect.pt')
        self.merge_params['neuralnetwork']['denoise']['filename'] = os.path.join(
            self.params['neural_nets_path'],
            'denoise.pt')

        self.merge_params['resources']['multi_processing'] = self.params['multi_processing']
        self.merge_params['resources']['n_processors'] = self.params['n_processors']
        self.merge_params['resources']['n_gpu_processors'] = self.params['n_gpu_processors']
        self.merge_params['resources']['n_sec_chunk'] = self.params['n_sec_chunk']
        self.merge_params['resources']['n_sec_chunk_gpu_detect'] = self.params['n_sec_chunk_gpu_detect']
        self.merge_params['resources']['n_sec_chunk_gpu_deconv'] = self.params['n_sec_chunk_gpu_deconv']
        self.merge_params['resources']['gpu_id'] = self.params['gpu_id']
        self.merge_params['resources']['generate_phy'] = self.params['generate_phy']
        self.merge_params['resources']['phy_percent_spikes'] = self.params['phy_percent_spikes']

        self.merge_params['recordings']['spatial_radius'] = self.params['spatial_radius']
        self.merge_params['recordings']['spike_size_ms'] = self.params['spike_size_ms']
        self.merge_params['recordings']['clustering_chunk'] = self.params['clustering_chunk']

        self.merge_params['deconvolution']['update_templates'] = self.params['update_templates']
        self.merge_params['deconvolution']['neuron_discover'] = self.params['neuron_discover']
        self.merge_params['deconvolution']['template_update_time'] = self.params['template_update_time']

    def _run(self, recording, output_folder):
        '''
        '''
        recording = recover_recording(recording)  # allows this to run on multiple jobs (not just multi-core)

        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        yass sort {config}
                    '''.format(config=os.path.join(output_folder, 'config.yaml'))
        else:
            shell_cmd = '''
                        #!/bin/bash
                        yass sort {config}
                    '''.format(config=os.path.join(output_folder, 'config.yaml'))

        shell_script = ShellScript(shell_cmd,
                                   script_path=os.path.join(output_folder, self.sorter_name),
                                   log_path=os.path.join(output_folder, self.sorter_name + '.log'),
                                   verbose=self.verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('yass returned a non-zero exit code')

    # Alessio might not want to put here;
    # better option to have a parameter "tune_nn" which
    def train(self, recording, output_folder):
        ''' Train NNs on yass prior to running yass sort
        '''
        print("TRAINING YASS (Note: using default spike width, neighbour chan radius; to change, see parameter files)")
        print("To use previously-trained NNs, change the NNs prior to running: ")
        print("            ss.set_NNs('path_to_NNs') (or set self.params['neural_nets_path'] = path_toNNs)")
        print("prior to running ss.run_sorter()")

        recording = recover_recording(recording)  # allows this to run on multiple jobs (not just multi-core)

        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        yass train {config}
                    '''.format(config=os.path.join(output_folder, 'config.yaml'))
        else:
            shell_cmd = '''
                        #!/bin/bash
                        yass train {config}
                    '''.format(config=os.path.join(output_folder, 'config.yaml'))

        shell_script = ShellScript(shell_cmd,
                                   script_path=os.path.join(output_folder, self.sorter_name),
                                   log_path=os.path.join(output_folder, self.sorter_name + '.log'),
                                   verbose=self.verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('yass returned a non-zero exit code')

        print("TRAINING COMPLETED. NNs located at: ", output_folder,
              "/tmp/nn_train/detect.pt and ",
              output_folder, "/tmp/nn_train/denoise.pt")

    def neural_nets_update_location(self, output_folder, neural_nets_path):
        ''' Update NNs to newly trained ones prior to running yass sort
        '''

        #################################################################
        #################### UPDATE CONFIG FILE TO NEW NNS ##############
        #################################################################
        self.merge_params['neuralnetwork']['denoise']['filename'] = os.path.join(
            neural_nets_path,
            'denoise.pt')

        self.merge_params['neuralnetwork']['detect']['filename'] = os.path.join(
            neural_nets_path,
            'detect.pt')

        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                    'config.yaml')

        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.merge_params, file)

    def neural_nets_default(self, output_folder):
        ''' Revert to default NNs
        '''
        #################################################################
        #################### UPDATE CONFIG FILE TO NEW NNS ##############
        #################################################################
        self.merge_params['neuralnetwork']['denoise']['filename'] = 'denoise.pt'
        self.merge_params['neuralnetwork']['detect']['filename'] = 'detect.pt'

        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                    'config.yaml')

        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.merge_params, file)

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.YassSortingExtractor(folder_path=Path(output_folder))
        return sorting
