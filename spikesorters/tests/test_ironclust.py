import unittest
import pytest
import spikeextractors as se
from spikesorters import IronclustSorter
from spikesorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not IronclustSorter.installed, reason='ironclust not installed')
class IronclustCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = IronclustSorter


if __name__ == '__main__':
    IronclustCommonTestSuite().test_on_toy()
    #~ IronclustCommonTestSuite().test_several_groups()