import unittest
import pytest
import spikeextractors as se
from spikesorters import Mountainsort4Sorter
from spikesorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
# @pytest.mark.skipif(True, reason='travis bug not fixed yet')
@pytest.mark.skipif(not Mountainsort4Sorter.is_installed(), reason='moutainsort4 not installed')
class Mountainsort4CommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Mountainsort4Sorter


if __name__ == '__main__':
    Mountainsort4CommonTestSuite().test_on_toy()
    Mountainsort4CommonTestSuite().test_several_groups()
    Mountainsort4CommonTestSuite().test_with_BinDatRecordingExtractor()
