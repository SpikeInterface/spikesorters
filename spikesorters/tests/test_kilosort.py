import os, getpass
if getpass.getuser() == 'samuel':
    kilosort_path = '/home/samuel/Documents/Spikeinterface/KiloSort/'
    os.environ["KILOSORT_PATH"] = kilosort_path


import unittest
import pytest
import spikeextractors as se
from spikesorters import KilosortSorter
from spikesorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not KilosortSorter.is_installed(), reason='kilosort not installed')
class KilosortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = KilosortSorter


if __name__ == '__main__':
    KilosortCommonTestSuite().test_on_toy()
    KilosortCommonTestSuite().test_several_groups()
    KilosortCommonTestSuite().test_with_BinDatRecordingExtractor()
