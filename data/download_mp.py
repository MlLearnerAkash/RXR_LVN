#!/usr/bin/env python3.8
# Downloads MP public data release
# Run with ./download_mp.py (or python download_mp.py on Windows)
# -*- coding: utf-8 -*-

import argparse
import os
import tempfile
import urllib.request

BASE_URL = 'http://kaldir.vc.in.tum.de/matterport/'
RELEASE = 'v1/scans'
RELEASE_TASKS = 'v1/tasks/'
RELEASE_SIZE = '1.3TB'
TOS_URL = BASE_URL + 'MP_TOS.pdf'
FILETYPES = [
    # 'cameras',
    # 'matterport_camera_intrinsics',
    # 'matterport_camera_poses',
    # 'matterport_color_images',
    # 'matterport_depth_images',
    # 'matterport_hdr_images',
    # 'matterport_mesh',
    'matterport_skybox_images',
    # 'undistorted_camera_parameters',
    # 'undistorted_color_images',
    # 'undistorted_depth_images',
    # 'undistorted_normal_images',
    # 'house_segmentations',
    # 'region_segmentations',
    # 'image_overlap_data',
    # 'poisson_meshes',
    # 'sens'
]
TASK_FILES = {
    'keypoint_matching_data': ['keypoint_matching/data.zip'],
    'keypoint_matching_models': ['keypoint_matching/models.zip'],
    'surface_normal_data': ['surface_normal/data_list.zip'],
    'surface_normal_models': ['surface_normal/models.zip'],
    'region_classification_data': ['region_classification/data.zip'],
    'region_classification_models': ['region_classification/models.zip'],
    'semantic_voxel_label_data': ['semantic_voxel_label/data.zip'],
    'semantic_voxel_label_models': ['semantic_voxel_label/models.zip'],
    'minos': ['mp3d_minos.zip'],
    'gibson': ['mp3d_for_gibson.tar.gz'],
    'habitat': ['mp3d_habitat.zip'],
    'pixelsynth': ['mp3d_pixelsynth.zip'],
    'igibson': ['mp3d_for_igibson.zip'],
    'mp360': [
        'mp3d_360/data_00.zip', 'mp3d_360/data_01.zip', 'mp3d_360/data_02.zip',
        'mp3d_360/data_03.zip', 'mp3d_360/data_04.zip', 'mp3d_360/data_05.zip',
        'mp3d_360/data_06.zip'
    ]
}

def get_release_scans(release_file_url):
    """
    Fetch the list of scan IDs from the release file.
    """
    with urllib.request.urlopen(release_file_url) as response:
        scans = [line.decode('utf-8').strip() for line in response]
    return scans


def download_file(url, out_file):
    """
    Download a single file if it does not already exist.
    """
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(out_file):
        print(f"    Downloading {url} -> {out_file}")
        # create a temporary file in the target directory
        fd, tmp_path = tempfile.mkstemp(dir=out_dir)
        os.close(fd)
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, out_file)
    else:
        print(f"WARNING: skipping existing file {out_file}")


def download_scan(scan_id, out_dir, file_types):
    """
    Download all specified file types for a given scan ID.
    """
    print(f"Downloading scan {scan_id}...")
    os.makedirs(out_dir, exist_ok=True)
    for ft in file_types:
        url = f"{BASE_URL}{RELEASE}/{scan_id}/{ft}.zip"
        out_file = os.path.join(out_dir, f"{ft}.zip")
        download_file(url, out_file)
    print(f"Completed scan {scan_id}.")


def download_release(release_scans, out_dir, file_types):
    """
    Download the entire release (all scans).
    """
    print(f"Downloading full MP release to {out_dir}...")
    for scan_id in release_scans:
        scan_out = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out, file_types)
    print("All scans downloaded.")


def download_task_data(task_keys, out_dir):
    """
    Download task-specific data and models.
    """
    print(f"Downloading task data for: {task_keys}...")
    for key in task_keys:
        if key not in TASK_FILES:
            print(f"ERROR: Unknown task data key: {key}")
            continue
        for part in TASK_FILES[key]:
            url = f"{BASE_URL}{RELEASE_TASKS}{part}"
            local_path = os.path.join(out_dir, part)
            download_file(url, local_path)
    print("Task data download complete.")


def main():
    parser = argparse.ArgumentParser(
        description="""
        Downloads the MP public data release.
        Examples:
          python download_mp.py -o base_dir --id ALL \
            --type cameras matterport_skybox_images \
            --task_data semantic_voxel_label_data
        """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-o', '--out_dir', required=True,
                        help='Base output directory for downloads')
    parser.add_argument('--task_data', nargs='+', default=[],
                        help='Task data keys to download: ' + ', '.join(TASK_FILES.keys()))
    parser.add_argument('--id', default='ALL',
                        help='Scan ID to download, or ALL for entire release')
    parser.add_argument('--type', nargs='+', choices=FILETYPES,
                        help='File types to download (default: all)')
    args = parser.parse_args()

    print('By continuing, you confirm acceptance of the MP Terms of Service:')
    print(TOS_URL)
    input('Press Enter to continue, or CTRL+C to exit...')

    # Fetch scan list
    release_file_url = f"{BASE_URL}{RELEASE}.txt"
    scans = get_release_scans(release_file_url)

    # Handle task data if requested
    if args.task_data:
        task_out = os.path.join(args.out_dir, RELEASE_TASKS)
        download_task_data(args.task_data, task_out)
        print()

    # Determine which file types to download
    file_types = args.type if args.type else FILETYPES

    # Download specific scan or full release
    if args.id.upper() != 'ALL':
        if args.id not in scans:
            print(f"ERROR: Scan ID '{args.id}' not found in release.")
        else:
            single_out = os.path.join(args.out_dir, RELEASE, args.id)
            download_scan(args.id, single_out, file_types)
    else:
        warning = (f"the entire release ({RELEASE_SIZE})" if len(file_types) == len(FILETYPES)
                   else f"all scans of types: {file_types}")
        print(f"WARNING: Downloading {warning}.")
        input('Press Enter to continue, or CTRL+C to exit...')
        full_out = os.path.join(args.out_dir, RELEASE)
        download_release(scans, full_out, file_types)


if __name__ == '__main__':
    main()
 