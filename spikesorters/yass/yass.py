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

    _default_params = {
        }

    _params_description = {
    }
       
    sorter_description = """Yass description; link to biorxiv"""

    installation_mesg = """\nTo use Yass run:\n
        >>> pip install yass-algorithm

        More information ...
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)
    
        config_default_location = '/home/cat/code/spikesorters_forked/spikesorters/yass/config_default.yaml'
        
        with open(config_default_location) as file:
            self.yass_params = yaml.load(file, Loader=yaml.FullLoader)
    
    @classmethod
    def is_installed(cls):
        return HAVE_YASS
    
    @staticmethod
    def get_sorter_version():
        return yass.__version__

    def _setup_recording(self, recording, output_folder):
        p = self.params
        source_dir = Path(output_folder).parent
        
        #################################################################
        #################### UPDATE ROOT FOLDER #########################
        #################################################################
        self.yass_params['data']['root_folder'] = str(output_folder)
        
        #################################################################
        #################### GEOMETRY FILE GENERATION ###################
        #################################################################
        probe_file_txt = os.path.join(output_folder,'geom.txt')
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
        # save binary file; THIS IS FROM KILOSORT
        input_file_path = os.path.join(output_folder, 'data.bin')
        
        recording.write_to_binary_dat_format(input_file_path, 
                                             dtype='int16', 
                                             chunk_mb=500) # time_axis=0,1 for C/F order
        
        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                   'config.yaml')
        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.yass_params, file)
        
        #################################################################
        ############ RUN NN TRAINING ON EXISTING DATASET ################
        #################################################################
        self.train(recording, output_folder)
        
        #################################################################
        ############ RUN NN TRAINING ON EXISTING DATASET ################
        #################################################################
        self.NNs_update_location(output_folder)


        #self.fname_config = fname_config
        
        # ALESSIO:
        # Expose more config file parameters that are sensiitive:
        #  e.g. spike width; smallest cluster; min firing rates;
        #  

    def _run(self, recording, output_folder):
        '''
        '''
        recording = recover_recording(recording)  # allows this to run on multiple jobs (not just multi-core)
        
        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        yass sort {config}
                    '''.format(config=os.path.join(output_folder,'config.yaml'))
        else:
            shell_cmd = '''
                        #!/bin/bash
                        yass sort {config}
                    '''.format(config=os.path.join(output_folder,'config.yaml'))

        shell_script = ShellScript(shell_cmd, 
                                   script_path=os.path.join(output_folder,self.sorter_name),
                                   log_path=os.path.join(output_folder,self.sorter_name+'.log'), 
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
        print ("TRAINING YASS (Note: using default spike width, radius etc.)")
        print ("To use previously-trained NNs, change the NNs prior to running: ")
        print ("            ss.set_NNs('path_to_NNs')")
        print ("prior to running ss.run_sorter()") 
        
        recording = recover_recording(recording)  # allows this to run on multiple jobs (not just multi-core)
        
        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        yass train {config}
                    '''.format(config=os.path.join(output_folder,'config.yaml'))
        else:
            shell_cmd = '''
                        #!/bin/bash
                        yass train {config}
                    '''.format(config=os.path.join(output_folder,'config.yaml'))

        shell_script = ShellScript(shell_cmd, 
                                   script_path=os.path.join(output_folder,self.sorter_name),
                                   log_path=os.path.join(output_folder,self.sorter_name+'.log'), 
                                   verbose=self.verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('yass returned a non-zero exit code')  
            
            
        print ("TRAINING COMPLETED. NNs located at: ", output_folder, 
                "/tmp/nn_train/detect.pt and ",
                output_folder,"/tmp/nn_train/denoise.pt")        
        
        
    def NNs_update_location(self, output_folder):
        ''' Update NNs to newly trained ones prior to running yass sort
        '''
        
        #################################################################
        #################### UPDATE CONFIG FILE TO NEW NNS ##############
        #################################################################
        self.yass_params['neuralnetwork']['denoise']['filename']= os.path.join(
                                            output_folder, 
                                            'tmp',
                                            'nn_train',
                                            'denoise.pt')
        self.yass_params['neuralnetwork']['detect']['filename']= os.path.join(
                                            output_folder, 
                                            'tmp',
                                            'nn_train',
                                            'detect.pt')
        
        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                   'config.yaml')
        
        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.yass_params, file)
        
        
    def NNs_default(self):
        ''' Revert to default NNs
        '''
        #################################################################
        #################### UPDATE CONFIG FILE TO NEW NNS ##############
        #################################################################
        self.yass_params['neuralnetwork']['denoise']['filename']= 'denoise.pt'
        self.yass_params['neuralnetwork']['detect']['filename']= 'detect.pt'
        
        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                   'config.yaml')
        
        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.yass_params, file)   
            
    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.SpykingCircusSortingExtractor(folder_path=Path(output_folder) / 'recording')
        return sorting
