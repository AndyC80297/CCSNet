import toml
import math 

from pathlib import Path
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict


def call_gwf_from_local(
    start,
    end,
    ifo="K1",
    frame_type="DATA",
    root_path=Path("/home/chiajui.chou/O4a/K1"),
    anchor_time = 1368959234,
    file_segment = 4096
):

    start_counter = file_segment * ((start - anchor_time) // file_segment) + anchor_time

    # Since (start - anchor_time) isn't alaway n * file_segment it's better to +1.
    num_call_file = math.ceil((end - start) / 4096) + 1
    
    gwf_files = [root_path /  f"{ifo[0]}-{ifo}_{frame_type}-{start_counter + i * file_segment}-{file_segment}.gwf" for i in range(num_call_file)]

    for file in gwf_files:

        try:
            my_abs_path = file.resolve(strict=True)
        except FileNotFoundError:
            raise "No file found"

    return gwf_files


def create_lcs(
    ifo,
    frametype,
    start_time,
    end_time,
    output_dir,
    urltype="file"
):
    """Select time, stateflag >>> get *.gwf file >>> 
    """
    head = "file://localhost"
    empty = ""
    
    files = find_urls(
        site=ifo[0],
        frametype=frametype,
        gpsstart=start_time,
        gpsend=end_time,
        urltype=urltype,
    )
    
    output_dir = output_dir / ifo
    output_dir.mkdir(parents=True, exist_ok=True)
    
    f = open(output_dir / "data_file.lcf", "a")
    for file in files:
        f.write(f"{file.replace(head, empty)}\n")
    f.close()

def create_lcs_from_local(
    ifo,
    frametype,
    start_time,
    end_time,
    output_dir,
    root_path=Path("/home/chiajui.chou/O4a/K1"),
    anchor_time = 1368959234,
    file_segment = 4096
):
    
    gwf_files = call_gwf_from_local(
        start=start_time,
        end=end_time,
        ifo=ifo,
        frame_type=frametype,
        root_path=root_path,
        anchor_time = anchor_time,
        file_segment = file_segment
    )

    output_dir = output_dir / ifo
    output_dir.mkdir(parents=True, exist_ok=True)

    f = open(output_dir / "data_file.lcf", "a")
    for file in gwf_files:
        f.write(f"{file}\n")
    f.close()