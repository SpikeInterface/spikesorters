import unittest
import pytest

from spikesorters import HerdingspikesSorter
from spikesorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
# @pytest.mark.skipif(True, reason='travis bug not fixed yet')
@pytest.mark.skipif(not HerdingspikesSorter.is_installed(), reason='herdingspikes not installed')
class HerdingspikesSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = HerdingspikesSorter


if __name__ == '__main__':
    HerdingspikesSorterCommonTestSuite().test_on_toy()
    HerdingspikesSorterCommonTestSuite().test_several_groups()
    HerdingspikesSorterCommonTestSuite().test_with_BinDatRecordingExtractor()
