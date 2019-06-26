
from .klusta import KlustaSorter
from .tridesclous import TridesclousSorter
from .mountainsort4 import Mountainsort4Sorter
from .ironclust import IronclustSorter
from .kilosort import KilosortSorter
from .kilosort2 import Kilosort2Sorter
from .spyking_circus import SpykingcircusSorter
from .herdingspikes import HerdingspikesSorter


sorter_full_list = [
    KlustaSorter,
    TridesclousSorter,
    Mountainsort4Sorter,
    IronclustSorter,
    KilosortSorter,
    Kilosort2Sorter,
    SpykingcircusSorter,
    HerdingspikesSorter
]

sorter_dict = {s.sorter_name: s for s in sorter_full_list}

installed_sorter_list = [s for s in sorter_full_list if s.installed]

def available_sorters():
    return sorted(list(sorter_dict.keys()))
