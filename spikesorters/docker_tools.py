import spikeextractors as se
import time
import numpy as np
from pathlib import Path

ss_folder = Path(__file__).parent

try:
    import hither2 as hither
    import docker

    HAVE_DOCKER = True

    default_docker_images = {
        "klusta": hither.DockerImageFromScript(name="klusta", dockerfile=str(ss_folder / "docker_images" / "v0.12" / "klusta" / "Dockerfile")),
        "mountainsort4": hither.DockerImageFromScript(name="ms4", dockerfile=str(ss_folder / "docker_images" / "v0.12" / "mountainsort4" / "Dockerfile")),
        "herdingspikes": hither.LocalDockerImage('spikeinterface/herdingspikes-si-0.12:0.3.7'),
        "spykingcircus": hither.LocalDockerImage('spikeinterface/spyking-circus-si-0.12:1.0.7')
    }

except ImportError:
    HAVE_DOCKER = False


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