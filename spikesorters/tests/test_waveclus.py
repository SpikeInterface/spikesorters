import unittest
import pytest
import spikeextractors as se
from spikesorters import WaveClusSorter
from spikesorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not WaveClusSorter.installed, reason='waveclus not installed')
class WaveClusCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = WaveClusSorter


if __name__ == '__main__':
    WaveClusCommonTestSuite().test_on_toy()
    #~ KilosortCommonTestSuite().test_several_groups()