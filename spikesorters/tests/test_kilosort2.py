import unittest
import pytest
import spikeextractors as se
from spikesorters import Kilosort2Sorter
from spikesorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not Kilosort2Sorter.installed, reason='kilosort not installed')
class Kilosort2CommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Kilosort2Sorter


if __name__ == '__main__':
    Kilosort2CommonTestSuite().test_on_toy()
    #~ KilosortCommonTestSuite().test_several_groups()