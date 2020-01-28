import os
import os, getpass
if getpass.getuser() == 'samuel':
    kilosort2_path = '/home/samuel/Documents/Spikeinterface/Kilosort2'
    os.environ["KILOSORT2_PATH"] = kilosort2_path

    kilosort_path = '/home/samuel/Documents/Spikeinterface/KiloSort/'
    os.environ["KILOSORT_PATH"] = kilosort_path

    ironclust_path = '/home/samuel/Documents/Spikeinterface/ironclust'
    os.environ["IRONCLUST_PATH"] = ironclust_path


import pytest

from spikesorters import print_sorter_version


def test_print_sorter_version():
    print_sorter_version()




if __name__ == '__main__':
    test_print_sorter_version()