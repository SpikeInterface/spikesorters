import spikeextractors as se
import spikesorters as ss
import numpy as np


def test_detection():
    rec, sort = se.example_datasets.toy_example(num_channels=4, duration=20, seed=0)

    # negative
    sort_d_n = ss.detect_spikes(rec)
    sort_dp_n = ss.detect_spikes(rec, parallel=True)

    assert 'channel' in sort_d_n.get_shared_unit_property_names()
    assert 'channel' in sort_dp_n.get_shared_unit_property_names()

    for u in sort_d_n.get_unit_ids():
        assert np.array_equal(sort_d_n.get_unit_spike_train(u), sort_dp_n.get_unit_spike_train(u))

    # positive
    sort_d_p = ss.detect_spikes(rec, detect_sign=1)
    sort_dp_p = ss.detect_spikes(rec, detect_sign=1, parallel=True)

    assert 'channel' in sort_d_p.get_shared_unit_property_names()
    assert 'channel' in sort_dp_p.get_shared_unit_property_names()

    for u in sort_d_p.get_unit_ids():
        assert np.array_equal(sort_d_p.get_unit_spike_train(u), sort_dp_p.get_unit_spike_train(u))

    # both
    sort_d_b = ss.detect_spikes(rec, detect_sign=0)
    sort_dp_b = ss.detect_spikes(rec, detect_sign=0, parallel=True)

    assert 'channel' in sort_d_b.get_shared_unit_property_names()
    assert 'channel' in sort_dp_b.get_shared_unit_property_names()

    for u in sort_d_b.get_unit_ids():
        assert np.array_equal(sort_d_b.get_unit_spike_train(u), sort_dp_b.get_unit_spike_train(u))


if __name__ == '__main__':
    test_detection()
