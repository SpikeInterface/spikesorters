import unittest
import pytest
import spikeextractors as se
from spikesorters import KilosortSorter
from spikesorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not KilosortSorter.installed, reason='kilosort not installed')
class KilosortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = KilosortSorter


if __name__ == '__main__':
    KilosortCommonTestSuite().test_on_toy()
    #~ KilosortCommonTestSuite().test_several_groups()