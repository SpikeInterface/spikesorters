from ..docker_tools import HAVE_DOCKER
from ..sorterlist import sorter_dict, sorter_full_list


if HAVE_DOCKER:
    # conditional definition of hither tools
    import time
    from pathlib import Path
    import hither2 as hither
    import spikeextractors as se
    import numpy as np
    import shutil
    from ..docker_tools import modify_input_folder, default_docker_images

    class SpikeSortingDockerHook(hither.RuntimeHook):
        def __init__(self):
            super().__init__()

        def precontainer(self, context: hither.PreContainerContext):
            # this gets run outside the container before the run, and we have a chance to mutate the kwargs,
            # add bind mounts, and set the image
            input_directory = context.kwargs['input_directory']
            output_directory = context.kwargs['output_directory']

            print("Input:", input_directory)
            print("Output:", output_directory)
            context.add_bind_mount(hither.BindMount(source=input_directory,
                                                    target='/input', read_only=True))
            context.add_bind_mount(hither.BindMount(source=output_directory,
                                                    target='/output', read_only=False))
            context.image = default_docker_images[context.kwargs['sorter_name']]
            context.kwargs['output_directory'] = '/output'
            context.kwargs['input_directory'] = '/input'


    @hither.function('run_sorter_docker_with_container',
                     '0.1.0',
                     image=True,
                     runtime_hooks=[SpikeSortingDockerHook()])
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
        assert recording.is_dumpable, "Cannot run not dumpable recordings in docker"
        if output_folder is None:
            output_folder = sorter_name + '_output'
        output_folder = Path(output_folder).absolute()
        output_folder.mkdir(exist_ok=True, parents=True)

        with hither.Config(use_container=True, show_console=True):
            dump_dict_container, input_directory = modify_input_folder(recording.dump_to_dict(), '/input')
            print(dump_dict_container)
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
        if delete_output_folder:
            shutil.rmtree(output_folder)
        return sorting
else:
    def _run_sorter_hither(sorter_name, recording, output_folder=None, delete_output_folder=False,
                           grouping_property=None, parallel=False, verbose=False, raise_error=True,
                           n_jobs=-1, joblib_backend='loky', **params):
        raise ImportError()


# generic launcher via function approach
def _run_sorter_local(sorter_name_or_class, recording, output_folder=None, delete_output_folder=False,
                      grouping_property=None, parallel=False, verbose=False, raise_error=True, n_jobs=-1,
                      joblib_backend='loky', **params):
    """
    Generic function to run a sorter via function approach.

    Two usages with name or class:

    by name:
       >>> sorting = run_sorter('tridesclous', recording)

    by class:
       >>> sorting = run_sorter(TridesclousSorter, recording)

    Parameters
    ----------
    sorter_name_or_class: str or SorterClass
        The sorter to retrieve default parameters from
    recording: RecordingExtractor
        The recording extractor to be spike sorted
    output_folder: str or Path
        Path to output folder
    delete_output_folder: bool
        If True, output folder is deleted (default False)
    grouping_property: str
        Splits spike sorting by 'grouping_property' (e.g. 'groups')
    parallel: bool
        If True and spike sorting is by 'grouping_property', spike sorting jobs are launched in parallel
    verbose: bool
        If True, output is verbose
    raise_error: bool
        If True, an error is raised if spike sorting fails (default). If False, the process continues and the error is
        logged in the log file.
    n_jobs: int
        Number of jobs when parallel=True (default=-1)
    joblib_backend: str
        joblib backend when parallel=True (default='loky')
    **params: keyword args
        Spike sorter specific arguments (they can be retrieved with 'get_default_params(sorter_name_or_class)'

    Returns
    -------
    sortingextractor: SortingExtractor
        The spike sorted data

    """
    if isinstance(sorter_name_or_class, str):
        SorterClass = sorter_dict[sorter_name_or_class]
    elif sorter_name_or_class in sorter_full_list:
        SorterClass = sorter_name_or_class
    else:
        raise ValueError('Unknown sorter')

    sorter = SorterClass(recording=recording, output_folder=output_folder, grouping_property=grouping_property,
                         verbose=verbose, delete_output_folder=delete_output_folder)
    sorter.set_params(**params)
    sorter.run(raise_error=raise_error, parallel=parallel, n_jobs=n_jobs, joblib_backend=joblib_backend)
    sortingextractor = sorter.get_result(raise_error=raise_error)

    return sortingextractor