from spikeextractors import YassSortingExtractor
from yass import YassSorter


# INITIALIZE A TOY DATASET
if __name__ == "__main__":
    
    import spikeextractors as se
    rec, sort = se.example_datasets.toy_example(duration=300)
    
    # INITIALIZE YASS SORTER OBJECT
    output_folder = '/media/cat/4TBSSD/spikeinterface/'
    sorter = YassSorter(recording=rec, output_folder=output_folder)
    
    # RUN SETUP DIRECTLY (Bypass BaseClass)
    sorter._setup_recording(rec=rec, output_folder=output_folder)

    
    # RUN TRAINING STEP
    sorter.train(rec, output_folder)

    # UPDATE NNs to TRAINED VERSIOn
    sorter.NNs_update()

    # RUN YASS
    sorter._run(rec, output_folder)

    # GET RESULTS
    from spikeextractors import YassSortingExtractor

    d = YassSortingExtractor(output_folder)

    unit_id = 0

    spikes = d.get_unit_spike_train(unit_id, start_frame=0, end_frame=40000)
    print ("Unit: ", unit_id, " spikes: ", spikes)

    temps = d.get_temps()
    print ("Templates: ", temps.shape)
    
