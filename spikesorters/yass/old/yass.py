import copy
from pathlib import Path
import os
import numpy as np
from numpy.lib.format import open_memmap
import sys
import yaml

import spikeextractors as se

from ..basesorter import BaseSorter
from ..sorter_tools import recover_recording

from ..utils.shellscript import ShellScript


from shellscript import ShellScript

try:
    import yass
    HAVE_YASS = True
except ImportError:
    HAVE_YASS = False

print ("HAVE_YASS: ", HAVE_YASS) 

#####################################################################
############# YASSSORTER CLASS ######################################
#####################################################################
class YassSorter(BaseSorter):
    """ 
    """

    sorter_name = 'yass'
    requires_locations = False
    verbose = False
    is_installed = HAVE_YASS

    # _default_params comes from default Yass config.yaml file; 
    #  Should be autoloaded from yass_installation_directory/samples/10chan/config.yaml
    #  - for now a copy is saved here: spikesorters/spikesorters/yass/config.yaml
    
#     _default_params = {
#         'adjacency_radius': 100,  # Channel neighborhood adjacency radius corresponding to geom file
#         'filter': True,
#         }

    # ALESSIO: Move thsi file locally to the yass directory;
    default_config_fname = './config_sample.yaml'
    with open(default_config_fname) as f:
        _default_params = yaml.load(f, Loader=yaml.FullLoader)   
    
    _params_description = {
        'data': " ...",
    }

    # 
    sorter_description = """Yass uses Neural Networks and SuperResolution Deconvolution to sort Retinal and 
                        cortical data. 
                        For more information see https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1
                        """

    installation_mesg = """\nTo Install and Use Yass follow the wiki: 
                            https://github.com/paninski-lab/yass/wiki
                        """

    def __init__(self, **kargs):
        print (**kargs)
        #BaseSorter.__init__(self, rec, output_folder)
        self.params = self._default_params
        
    @classmethod
    def is_installed(cls):
        #return HAVE_YASS
        #return check_if_installed(cls.kilosort2_path)
        return self.HAVE_YASS
    
    @staticmethod
    def get_sorter_version():
        return yass.__version__

    # NEED TO CHANGE THIS FOR YASS ALSO
    # n_chan = recording.get_num_channels()
    # n_frames = recording.get_num_frames()
    # chunk_size = 2 ** 24 // n_chan
    
    # also make a default config file for
    # https://github.com/SpikeInterface/spikesorters/blob/master/spikesorters/spyking_circus/config_default.params
    # {} reserved for params that need to be updated at run time;
    
    # this function parses params and creates config file and binary data and geometry file also
    # and saving everything to the output folder
    # Cat: this function changes the minimum required default values; 
    def _setup_recording(self, recording, output_folder):
        p = self.params
        source_dir = Path(output_folder).parent

        #################################################################
        #################### UPDATE ROOT FOLDER #########################
        #################################################################
        # float(self._recording.sample_rate.rescale('Hz').magnitude)
        self.params['data']['root_folder'] = output_folder
        #self.params['data']['geometry'] = 'geom.csv'
        
        #################################################################
        #################### GEOMETRY FILE GENERATION ###################
        #################################################################
        # save prb file
        # note: only one group here, the split is done in basesorter
        probe_file_csv = os.path.join(output_folder,'geom.csv')
        probe_file_txt = os.path.join(output_folder,'geom.txt')
        # ALESSIO .saveto probe saved .prb file; have to d thisourselves.
        #  
        adjacency_radius = -1
        recording.save_to_probe_file(probe_file_csv, 
                                     grouping_property=None,
                                     radius=adjacency_radius)
        
        import csv

        with open(probe_file_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            geom_txt = np.float32(np.vstack(csv_reader))
            np.savetxt(probe_file_txt, geom_txt)
        
        #################################################################
        #################### UPDATE SAMPLING RATE #######################
        #################################################################
        # float(self._recording.sample_rate.rescale('Hz').magnitude)
        self.params['recordings']['sampling_rate'] = recording.get_sampling_frequency()
        
        
        #################################################################
        #################### UPDATE N_CHAN  #############################
        #################################################################
        self.params['recordings']['n_channels'] = recording.get_num_channels()
        
        
        #################################################################
        #################### SAVE RAW INT16 data ########################
        #################################################################
        # ALESSIO Look at Kilosort 
        # There is alrady an extractor se.Mea1kRecordingExtractor()
        # all the functions are there already to concatenate in time;
        # multi-recording time extractor;
        
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
            documents = yaml.dump(self.params, file)
        
        #self.fname_config = fname_config
        
        # ALESSIO:
        # Expose more config file parameters that are sensiitive:
        #  e.g. spike width; smallest cluster; min firing rates;
        #  
            
    # FUNCTION TO RUN YASS 
    #def _run(self,recording, output_folder):  # SOMETIMES want to access more information from recording in
                                               # this step
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
        '''
        '''
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
            
    def NNs_update(self):
        ''' Update NNs to newly trained ones
        '''
        
        #################################################################
        #################### UPDATE CONFIG FILE TO NEW NNS ##############
        #################################################################
        self.params['neuralnetwork']['denoise']['filename']= os.path.join(
                                            output_folder, 
                                            'tmp',
                                            'nn_train',
                                            'denoise.pt')
        self.params['neuralnetwork']['detect']['filename']= os.path.join(
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
            documents = yaml.dump(self.params, file)
        
        
    def NNs_default(self):
        ''' Revert to default NNs
        '''
        #################################################################
        #################### UPDATE CONFIG FILE TO NEW NNS ##############
        #################################################################
        self.params['neuralnetwork']['denoise']['filename']= 'denoise.pt'
        self.params['neuralnetwork']['detect']['filename']= 'detect.pt'
        
        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                   'config.yaml')
        
        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.params, file)        
        
    
    # 
    @staticmethod
    def get_result_from_folder(output_folder):
        #sorting = se.YassSortingExtractor(folder_path=Path(output_folder) / 'tmp/output/spike_train.npy')
        sorting = se.YassSortingExtractor(folder_path=Path(output_folder))
        return sorting
    
