from .sorterlist import _run_sorter_local
import spikeextractors as se
import time
import numpy as np
from pathlib import Path

try:
    import hither2 as hither
    import docker

    HAVE_DOCKER = True

    default_docker_images = {
        "klusta": hither.LocalDockerImage('spikeinterface/klusta-si-0.12:0.2.7'),
        "mountainsort4": hither.LocalDockerImage('spikeinterface/mountainsort4-si-0.12:1.0.0'),
        "herdingspikes": hither.LocalDockerImage('spikeinterface/herdingspikes-si-0.12:0.3.7'),
        "spykingcircus": hither.LocalDockerImage('spikeinterface/spyking-circus-si-0.12:1.0.7')
    }

    class SpikeSortingDockerHook(hither.RuntimeHook):
        def __init__(self):
            super().__init__()

        def precontainer(self, context: hither.PreContainerContext):
            # this gets run outside the container before the run, and we have a chance to mutate the kwargs,
            # add bind mounts, and set the image
            input_directory = context.kwargs['input_directory']
            output_directory = context.kwargs['output_directory']

            context.add_bind_mount(hither.BindMount(source=input_directory,
                                                    target='/input', read_only=True))
            context.add_bind_mount(hither.BindMount(source=output_directory,
                                                    target='/output', read_only=False))
            context.image = default_docker_images[context.kwargs['sorter_name']]
            context.kwargs['output_directory'] = '/output'
            context.kwargs['input_directory'] = '/input'


    @hither.function('run_sorter_docker_with_container', '0.1.0', image=True, runtime_hooks=[SpikeSortingDockerHook()])
    def run_sorter_docker_with_container(
            recording_dict, sorter_name, input_directory, output_directory, **kwargs
    ):
        recording = se.load_extractor_from_dict(recording_dict)
        # run sorter
        kwargs["output_folder"] = f"{output_directory}/working"
        t_start = time.time()
        # set output folder within the container
        sorting = _run_sorter_local(sorter_name, recording, **kwargs)
        t_stop = time.time()
        print(f'{sorter_name} run time {np.round(t_stop - t_start)}s')
        # save sorting to npz
        se.NpzSortingExtractor.write_sorting(sorting, f"{output_directory}/sorting_docker.npz")

    def _run_sorter_hither(sorter_name, recording, output_folder=None, delete_output_folder=False,
                           grouping_property=None, parallel=False, verbose=False, raise_error=True,
                           n_jobs=-1, joblib_backend='loky', **params):
        output_folder = Path(output_folder).absolute()
        output_folder.mkdir(exist_ok=True, parents=True)

        with hither.Config(use_container=True, show_console=True):
            dump_dict_container, input_directory = modify_input_folder(recording.dump_to_dict(), '/input')
            kwargs = dict(recording_dict=dump_dict_container,
                          sorter_name=sorter_name,
                          output_folder=str(output_folder),
                          delete_output_folder=False,
                          grouping_property=grouping_property, parallel=parallel,
                          verbose=verbose, raise_error=raise_error, n_jobs=n_jobs,
                          joblib_backend=joblib_backend)

            kwargs.update(params)
            kwargs.update({'input_directory': str(input_directory), 'output_directory': str(output_folder)})
            sorting_job = hither.Job(run_sorter_docker_with_container, kwargs)
            sorting_job.wait()
        sorting = se.NpzSortingExtractor(output_folder / "sorting_docker.npz")
        return sorting

except ImportError:
    HAVE_DOCKER = False
    def _run_sorter_hither(sorter_name, recording, output_folder=None, delete_output_folder=False,
                           grouping_property=None, parallel=False, verbose=False, raise_error=True,
                           n_jobs=-1, joblib_backend='loky', **params):
        raise NotImplementedError


def modify_input_folder(dump_dict, input_folder="/input"):
    if "kwargs" in dump_dict.keys():
        dcopy_kwargs, folder_to_mount = modify_input_folder(dump_dict["kwargs"])
        dump_dict["kwargs"] = dcopy_kwargs
        return dump_dict, folder_to_mount
    else:
        if "file_path" in dump_dict:
            file_path = Path(dump_dict["file_path"])
            folder_to_mount = file_path.parent
            file_relative = file_path.relative_to(folder_to_mount)
            dump_dict["file_path"] = f"{input_folder}/{str(file_relative)}"
            return dump_dict, folder_to_mount
        elif "folder_path" in dump_dict:
            folder_path = Path(dump_dict["folder_path"])
            folder_to_mount = folder_path.parent
            folder_relative = folder_path.relative_to(folder_to_mount)
            dump_dict["folder_path"] = f"{input_folder}/{str(folder_relative)}"
            return dump_dict, folder_to_mount
        elif "file_or_folder_path" in dump_dict:
            file_or_folder_path = Path(dump_dict["file_or_folder_path"])
            folder_to_mount = file_or_folder_path.parent
            file_or_folder_relative = file_or_folder_path.relative_to(folder_to_mount)
            dump_dict["file_or_folder_path"] = f"{input_folder}/{str(file_or_folder_relative)}"
            return dump_dict, folder_to_mount
        else:
            raise Exception


def return_local_data_folder(recording, input_folder='/input'):
    """
    Modifies recording dictionary so that the file_path, folder_path, or file_or_folder path is relative to the
    'input_folder'

    Parameters
    ----------
    recording: se.RecordingExtractor
    input_folder: str

    Returns
    -------
    dump_dict: dict

    """
    assert recording.is_dumpable
    from copy import deepcopy

    d = recording.dump_to_dict()
    dcopy = deepcopy(d)

    return modify_input_folder(dcopy, input_folder)