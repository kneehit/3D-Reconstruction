#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kneehit
"""


#%%

from opensfm import dataset
from opensfm import exif
import logging

import copy

import time 
from timeit import default_timer as timer
from opensfm.context import parallel_map
from opensfm import io
from opensfm import log
from opensfm import dense

import numpy as np

from opensfm import matching

from opensfm import tracking
from opensfm import reconstruction

from opensfm import mesh
from opensfm import types



from utils import _extract_exif, detect, detect_features_report, match_features_report, tracks_report 
from utils import undistort_reconstruction
dataset_path = 'dataset/flash'
log.setup()

logger = logging.getLogger('reconstruction')
logging.getLogger("reconstruction").setLevel(logging.INFO)

#%%

def extract_metadata( dataset_path):
    start = time.time()
    data = dataset.DataSet(dataset_path)

    exif_overrides = {}
    if data.exif_overrides_exists():
        exif_overrides = data.load_exif_overrides()

    camera_models = {}
    for image in data.images():
        if data.exif_exists(image):
            logging.info('Loading existing EXIF for {}'.format(image))
            d = data.load_exif(image)
        else:
            logging.info('Extracting EXIF for {}'.format(image))
            d = _extract_exif(image, data)

            if image in exif_overrides:
                d.update(exif_overrides[image])

            data.save_exif(image, d)

        if d['camera'] not in camera_models:
            camera = exif.camera_from_exif_metadata(d, data)
            camera_models[d['camera']] = camera

    # Override any camera specified in the camera models overrides file.
    if data.camera_models_overrides_exists():
        overrides = data.load_camera_models_overrides()
        if "all" in overrides:
            for key in camera_models:
                camera_models[key] = copy.copy(overrides["all"])
                camera_models[key].id = key
        else:
            for key, value in overrides.items():
                camera_models[key] = value
    data.save_camera_models(camera_models)

    end = time.time()
    with open(data.profile_log(), 'a') as fout:
        fout.write('extract_metadata: {0}\n'.format(end - start))


def detect_features(dataset_path):
    data = dataset.DataSet(dataset_path)
    images = data.images()

    arguments = [(image, data) for image in images]

    start = timer()
    processes = data.config['processes']
    parallel_map(detect, arguments, processes, 1)
    end = timer()
    with open(data.profile_log(), 'a') as fout:
        fout.write('detect_features: {0}\n'.format(end - start))

    detect_features_report(data, end - start)

def match_features(dataset_path):
    data = dataset.DataSet(dataset_path)
    images = data.images()

    start = timer()
    pairs_matches, preport = matching.match_images(data, images, images)
    matching.save_matches(data, images, pairs_matches)
    end = timer()

    with open(data.profile_log(), 'a') as fout:
        fout.write('match_features: {0}\n'.format(end - start))
    match_features_report(data, preport, list(pairs_matches.keys()), end - start)


def create_tracks(dataset_path):
    data = dataset.DataSet(dataset_path)

    start = timer()
    features_tracks, colors = tracking.load_features(data, data.images())
    features_end = timer()
    matches = tracking.load_matches(data, data.images())
    matches_end = timer()
    tracks_manager = tracking.create_tracks_manager(features_tracks, colors, matches,
                                                    data.config)
    tracks_end = timer()
    data.save_tracks_manager(tracks_manager)
    end = timer()

    with open(data.profile_log(), 'a') as fout:
        fout.write('create_tracks: {0}\n'.format(end - start))

    tracks_report(data,
                      tracks_manager,
                      features_end - start,
                      matches_end - features_end,
                      tracks_end - matches_end)
    
    
def reconstruct(dataset_path):
    start = time.time()
    data = dataset.DataSet(dataset_path)
    tracks_manager = data.load_tracks_manager()
    report, reconstructions = reconstruction.\
        incremental_reconstruction(data, tracks_manager)
    end = time.time()
    with open(data.profile_log(), 'a') as fout:
        fout.write('reconstruct: {0}\n'.format(end - start))
    data.save_reconstruction(reconstructions)
    data.save_report(io.json_dumps(report), 'reconstruction.json')
    
    
def create_mesh(dataset_path):
    start = time.time()
    data = dataset.DataSet(dataset_path)
    tracks_manager = data.load_tracks_manager()
    reconstructions = data.load_reconstruction()

    all_shot_ids = set(tracks_manager.get_shot_ids())
    for i, r in enumerate(reconstructions):
        for shot in r.shots.values():
            if shot.id in all_shot_ids:
                vertices, faces = mesh.triangle_mesh(
                    shot.id, r, tracks_manager, data)
                shot.mesh = types.ShotMesh()
                shot.mesh.vertices = vertices
                shot.mesh.faces = faces

    data.save_reconstruction(reconstructions,
                             filename='reconstruction.meshed.json',
                             minify=True)

    end = time.time()
    with open(data.profile_log(), 'a') as fout:
        fout.write('mesh: {0}\n'.format(end - start))
        
def undistort(dataset_path,reconstruction = None,tracks = None,reconstruction_index = 0, 
        output = 'undistorted',):
    data = dataset.DataSet(dataset_path)
    udata = dataset.UndistortedDataSet(data, output)
    reconstructions = data.load_reconstruction(reconstruction)
    if data.tracks_exists(tracks):
        tracks_manager = data.load_tracks_manager(tracks)
    else:
        tracks_manager = None

    if reconstructions:
        r = reconstructions[reconstruction_index]
        undistort_reconstruction(tracks_manager, r, data, udata)
        
def compute_depthmaps(dataset_path, subfolder = 'undistorted', interactive = False):
    data = dataset.DataSet(dataset_path)
    udata = dataset.UndistortedDataSet(data, subfolder)
    data.config['interactive'] = interactive
    reconstructions = udata.load_undistorted_reconstruction()
    tracks_manager = udata.load_undistorted_tracks_manager()

    dense.compute_depthmaps(udata, tracks_manager, reconstructions[0])
#%%


def main(dataset_path):

    extract_metadata(dataset_path)
    
    
    
    detect_features(dataset_path)
    
    
    match_features(dataset_path)
    
    
    create_tracks(dataset_path)
    
    reconstruct(dataset_path)
    
    create_mesh(dataset_path)
    
    
    undistort(dataset_path)
    
    
    compute_depthmaps(dataset_path)


main(dataset_path)





