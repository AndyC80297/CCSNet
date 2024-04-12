import toml

from pathlib import Path
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict


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