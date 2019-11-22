import spikeextractors as se
import spikesorters as ss
import numpy as np


def test_detection():
    rec, sort  = se.example_datasets.toy_example(num_channels=4, duration=20)

    sort_d = ss.detect_spikes(rec)
    sort_dp = ss.detect_spikes(rec, parallel=True)

    assert 'channel' in sort_d.get_shared_unit_property_names()
    assert 'channel' in sort_dp.get_shared_unit_property_names()

    for u in sort_d.get_unit_ids():
        assert np.array_equal(sort_d.get_unit_spike_train(u), sort_dp.get_unit_spike_train(u))
