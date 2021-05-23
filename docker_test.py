import spikeextractors as se
import spikesorters as ss


rec, _ = se.example_datasets.toy_example(dumpable=True)

output_folder = "ms4_test_docker"

sorting = ss.run_klusta(rec, output_folder=output_folder, use_docker=True)

print(f"KL found #{len(sorting.get_unit_ids())} units")


# output_folder = "kl_test_docker"
#
# sorting_KL = ssd.run_klusta(rec, output_folder=output_folder)
#
# print(f"KL found #{len(sorting_KL.get_unit_ids())} units")
#
# rec, _ = se.example_datasets.toy_example(dumpable=True)
#
# output_folder = "sc_test_docker"
#
# sorting_SC = ssd.run_spykingcircus(rec, output_folder=output_folder)
#
# print(f"SC found #{len(sorting_SC.get_unit_ids())} units")
#
# rec, _ = se.example_datasets.toy_example(dumpable=True)
#
# output_folder = "hs_test_docker"
#
# sorting_HS = ssd.run_herdingspikes(rec, output_folder=output_folder)
#
# print(f"HS found #{len(sorting_HS.get_unit_ids())} units")
