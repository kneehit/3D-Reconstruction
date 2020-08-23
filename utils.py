#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
from timeit import default_timer as timer

import numpy as np
import cv2

from opensfm import exif

from opensfm import bow
from opensfm import dataset
from opensfm import features
from opensfm import io
from opensfm import log
from opensfm.context import parallel_map
from opensfm import transformations as transf     
from opensfm import types
from opensfm import pysfm
from opensfm import pygeometry


logger = logging.getLogger('reconstruction')


def _extract_exif(image, data):
     # EXIF data in Image
    d = exif.extract_exif_from_file(data.open_image_file(image))

    # Image Height and Image Width
    if d['width'] <= 0 or not data.config['use_exif_size']:
        d['height'], d['width'] = data.image_size(image)

    d['camera'] = exif.camera_id(d)

    return d


def detect(args):
    image, data = args

    log.setup()

    need_words = data.config['matcher_type'] == 'WORDS' or data.config['matching_bow_neighbors'] > 0
    has_words = not need_words or data.words_exist(image)
    has_features = data.features_exist(image)

    if has_features and has_words:
        logger.info('Skip recomputing {} features for image {}'.format(
            data.feature_type().upper(), image))
        return

    logger.info('Extracting {} features for image {}'.format(
        data.feature_type().upper(), image))

    start = timer()

    p_unmasked, f_unmasked, c_unmasked = features.extract_features(
        data.load_image(image), data.config)

    fmask = data.load_features_mask(image, p_unmasked)

    p_unsorted = p_unmasked[fmask]
    f_unsorted = f_unmasked[fmask]
    c_unsorted = c_unmasked[fmask]

    if len(p_unsorted) == 0:
        logger.warning('No features found in image {}'.format(image))
        return

    size = p_unsorted[:, 2]
    order = np.argsort(size)
    p_sorted = p_unsorted[order, :]
    f_sorted = f_unsorted[order, :]
    c_sorted = c_unsorted[order, :]
    data.save_features(image, p_sorted, f_sorted, c_sorted)

    if need_words:
        bows = bow.load_bows(data.config)
        n_closest = data.config['bow_words_to_match']
        closest_words = bows.map_to_words(
            f_sorted, n_closest, data.config['bow_matcher_type'])
        data.save_words(image, closest_words)

    end = timer()
    report = {
        "image": image,
        "num_features": len(p_sorted),
        "wall_time": end - start,
    }
    data.save_report(io.json_dumps(report), 'features/{}.json'.format(image))


def undistort_reconstruction(tracks_manager, reconstruction, data, udata):
    urec = types.Reconstruction()
    urec.points = reconstruction.points
    utracks_manager = pysfm.TracksManager()

    logger.debug('Undistorting the reconstruction')
    undistorted_shots = {}
    for shot in reconstruction.shots.values():
        if shot.camera.projection_type == 'perspective':
            camera = perspective_camera_from_perspective(shot.camera)
            subshots = [get_shot_with_different_camera(shot, camera)]
        elif shot.camera.projection_type == 'brown':
            camera = perspective_camera_from_brown(shot.camera)
            subshots = [get_shot_with_different_camera(shot, camera)]
        elif shot.camera.projection_type == 'fisheye':
            camera = perspective_camera_from_fisheye(shot.camera)
            subshots = [get_shot_with_different_camera(shot, camera)]
        elif shot.camera.projection_type in ['equirectangular', 'spherical']:
            subshot_width = int(data.config['depthmap_resolution'])
            subshots = perspective_views_of_a_panorama(shot, subshot_width)

        for subshot in subshots:
            urec.add_camera(subshot.camera)
            urec.add_shot(subshot)
            if tracks_manager:
                add_subshot_tracks(tracks_manager, utracks_manager, shot, subshot)
        undistorted_shots[shot.id] = subshots

    udata.save_undistorted_reconstruction([urec])
    if tracks_manager:
        udata.save_undistorted_tracks_manager(utracks_manager)

    arguments = []
    for shot in reconstruction.shots.values():
        arguments.append((shot, undistorted_shots[shot.id], data, udata))

    processes = data.config['processes']
    parallel_map(undistort_image_and_masks, arguments, processes)


def undistort_image_and_masks(arguments):
    shot, undistorted_shots, data, udata = arguments
    log.setup()
    logger.debug('Undistorting image {}'.format(shot.id))

    # Undistort image
    image = data.load_image(shot.id, unchanged=True, anydepth=True)
    if image is not None:
        max_size = data.config['undistorted_image_max_size']
        undistorted = undistort_image(shot, undistorted_shots, image,
                                      cv2.INTER_AREA, max_size)
        for k, v in undistorted.items():
            udata.save_undistorted_image(k, v)

    # Undistort mask
    mask = data.load_mask(shot.id)
    if mask is not None:
        undistorted = undistort_image(shot, undistorted_shots, mask,
                                      cv2.INTER_NEAREST, 1e9)
        for k, v in undistorted.items():
            udata.save_undistorted_mask(k, v)

    # Undistort segmentation
    segmentation = data.load_segmentation(shot.id)
    if segmentation is not None:
        undistorted = undistort_image(shot, undistorted_shots, segmentation,
                                      cv2.INTER_NEAREST, 1e9)
        for k, v in undistorted.items():
            udata.save_undistorted_segmentation(k, v)

    # Undistort detections
    detection = data.load_detection(shot.id)
    if detection is not None:
        undistorted = undistort_image(shot, undistorted_shots, detection,
                                      cv2.INTER_NEAREST, 1e9)
        for k, v in undistorted.items():
            udata.save_undistorted_detection(k, v)


def undistort_image(shot, undistorted_shots, original, interpolation,
                    max_size):
    """Undistort an image into a set of undistorted ones.

    Args:
        shot: the distorted shot
        undistorted_shots: the set of undistorted shots covering the
            distorted shot field of view. That is 1 for most camera
            types and 6 for equirectangular cameras.
        original: the original distorted image array.
        interpolation: the opencv interpolation flag to use.
        max_size: maximum size of the undistorted image.
    """
    if original is None:
        return

    projection_type = shot.camera.projection_type
    if projection_type in ['perspective', 'brown', 'fisheye']:
        new_camera = undistorted_shots[0].camera
        height, width = original.shape[:2]
        map1, map2 = pygeometry.compute_camera_mapping(shot.camera, new_camera, width, height)
        undistorted = cv2.remap(original, map1, map2, interpolation)
        return {shot.id: scale_image(undistorted, max_size)}
    elif projection_type in ['equirectangular', 'spherical']:
        subshot_width = undistorted_shots[0].camera.width
        width = 4 * subshot_width
        height = width // 2
        image = cv2.resize(original, (width, height), interpolation=interpolation)
        mint = cv2.INTER_LINEAR if interpolation == cv2.INTER_AREA else interpolation
        res = {}
        for subshot in undistorted_shots:
            undistorted = render_perspective_view_of_a_panorama(
                image, shot, subshot, mint)
            res[subshot.id] = scale_image(undistorted, max_size)
        return res
    else:
        raise NotImplementedError(
            'Undistort not implemented for projection type: {}'.format(
                shot.camera.projection_type))


def scale_image(image, max_size):
    """Scale an image not to exceed max_size."""
    height, width = image.shape[:2]
    factor = max_size / float(max(height, width))
    if factor >= 1:
        return image
    width = int(round(width * factor))
    height = int(round(height * factor))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def get_shot_with_different_camera(shot, camera):
    """Copy shot and replace camera."""
    ushot = types.Shot()
    ushot.id = shot.id
    ushot.camera = camera
    ushot.pose = shot.pose
    ushot.metadata = shot.metadata
    return ushot


def perspective_camera_from_perspective(distorted):
    """Create an undistorted camera from a distorted."""
    camera = pygeometry.Camera.create_perspective(distorted.focal, 0.0, 0.0)
    camera.id = distorted.id
    camera.width = distorted.width
    camera.height = distorted.height
    return camera


def perspective_camera_from_brown(brown):
    """Create a perspective camera froma a Brown camera."""
    camera = pygeometry.Camera.create_perspective(
        brown.focal * (1 + brown.aspect_ratio) / 2.0, 0.0, 0.0)
    camera.id = brown.id
    camera.width = brown.width
    camera.height = brown.height
    return camera


def perspective_camera_from_fisheye(fisheye):
    """Create a perspective camera from a fisheye."""
    camera = pygeometry.Camera.create_perspective(fisheye.focal, 0.0, 0.0)
    camera.id = fisheye.id
    camera.width = fisheye.width
    camera.height = fisheye.height
    return camera


def perspective_views_of_a_panorama(spherical_shot, width):
    """Create 6 perspective views of a panorama."""
    camera = pygeometry.Camera.create_perspective(0.5, 0.0, 0.0)
    camera.id = 'perspective_panorama_camera'
    camera.width = width
    camera.height = width

    names = ['front', 'left', 'back', 'right', 'top', 'bottom']
    rotations = [
        transf.rotation_matrix(-0 * np.pi / 2, (0, 1, 0)),
        transf.rotation_matrix(-1 * np.pi / 2, (0, 1, 0)),
        transf.rotation_matrix(-2 * np.pi / 2, (0, 1, 0)),
        transf.rotation_matrix(-3 * np.pi / 2, (0, 1, 0)),
        transf.rotation_matrix(-np.pi / 2, (1, 0, 0)),
        transf.rotation_matrix(+np.pi / 2, (1, 0, 0)),
    ]
    shots = []
    for name, rotation in zip(names, rotations):
        shot = types.Shot()
        shot.id = '{}_perspective_view_{}'.format(spherical_shot.id, name)
        shot.camera = camera
        R = np.dot(rotation[:3, :3], spherical_shot.pose.get_rotation_matrix())
        o = spherical_shot.pose.get_origin()
        shot.pose = types.Pose()
        shot.pose.set_rotation_matrix(R)
        shot.pose.set_origin(o)
        shots.append(shot)
    return shots


def render_perspective_view_of_a_panorama(image, panoshot, perspectiveshot,
                                          interpolation=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_WRAP):
    """Render a perspective view of a panorama."""
    # Get destination pixel coordinates
    dst_shape = (perspectiveshot.camera.height, perspectiveshot.camera.width)
    dst_y, dst_x = np.indices(dst_shape).astype(np.float32)
    dst_pixels_denormalized = np.column_stack([dst_x.ravel(), dst_y.ravel()])

    dst_pixels = features.normalized_image_coordinates(
        dst_pixels_denormalized,
        perspectiveshot.camera.width,
        perspectiveshot.camera.height)

    # Convert to bearing
    dst_bearings = perspectiveshot.camera.pixel_bearing_many(dst_pixels)

    # Rotate to panorama reference frame
    rotation = np.dot(panoshot.pose.get_rotation_matrix(),
                      perspectiveshot.pose.get_rotation_matrix().T)
    rotated_bearings = np.dot(dst_bearings, rotation.T)

    # Project to panorama pixels
    src_pixels = panoshot.camera.project_many(rotated_bearings)
    src_pixels_denormalized = features.denormalized_image_coordinates(
        src_pixels, image.shape[1], image.shape[0])

    src_pixels_denormalized.shape = dst_shape + (2,)

    # Sample color
    x = src_pixels_denormalized[..., 0].astype(np.float32)
    y = src_pixels_denormalized[..., 1].astype(np.float32)
    colors = cv2.remap(image, x, y, interpolation, borderMode=borderMode)

    return colors


def add_subshot_tracks(tracks_manager, utracks_manager, shot, subshot):
    """Add shot tracks to the undistorted tracks_manager."""
    if shot.id not in tracks_manager.get_shot_ids():
        return

    if shot.camera.projection_type in ['equirectangular', 'spherical']:
        add_pano_subshot_tracks(tracks_manager, utracks_manager, shot, subshot)
    else:
        for track_id, obs in tracks_manager.get_shot_observations(shot.id).items():
            utracks_manager.add_observation(subshot.id, track_id, obs)


def add_pano_subshot_tracks(tracks_manager, utracks_manager, panoshot, perspectiveshot):
    """Add edges between subshots and visible tracks."""
    for track_id, obs in tracks_manager.get_shot_observations(panoshot.id).items():
        bearing = panoshot.camera.pixel_bearing(obs.point)
        rotation = np.dot(perspectiveshot.pose.get_rotation_matrix(),
                          panoshot.pose.get_rotation_matrix().T)

        rotated_bearing = np.dot(bearing, rotation.T)
        if rotated_bearing[2] <= 0:
            continue

        perspective_feature = perspectiveshot.camera.project(rotated_bearing)
        if (perspective_feature[0] < -0.5 or
                perspective_feature[0] > 0.5 or
                perspective_feature[1] < -0.5 or
                perspective_feature[1] > 0.5):
            continue

        obs.point = perspective_feature
        utracks_manager.add_observation(perspectiveshot.id, track_id, obs)    
    
def detect_features_report(data, wall_time):
    image_reports = []
    for image in data.images():
        try:
            txt = data.load_report('features/{}.json'.format(image))
            image_reports.append(io.json_loads(txt))
        except IOError:
            logger.warning('No feature report image {}'.format(image))

    report = {
        "wall_time": wall_time,
        "image_reports": image_reports
    }
    data.save_report(io.json_dumps(report), 'features.json')
    
def match_features_report(data, preport, pairs, wall_time):
    report = {
        "wall_time": wall_time,
        "num_pairs": len(pairs),
        "pairs": pairs,
    }
    report.update(preport)
    data.save_report(io.json_dumps(report), 'matches.json')

def tracks_report(data, tracks_manager,
                 features_time, matches_time, tracks_time):
    view_graph = [(k[0], k[1], v) for k, v in tracks_manager.get_all_pairs_connectivity().items()]

    report = {
        "wall_times": {
            "load_features": features_time,
            "load_matches": matches_time,
            "compute_tracks": tracks_time,
        },
        "wall_time": features_time + matches_time + tracks_time,
        "num_images": tracks_manager.num_shots(),
        "num_tracks": tracks_manager.num_tracks(),
        "view_graph": view_graph
    }
    data.save_report(io.json_dumps(report), 'tracks.json')
