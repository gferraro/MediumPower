"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import time
import yaml
from datetime import datetime

import logging
from motiondetector import WeightedBackground


class ClipTrackExtractor:
    PREVIEW = "preview"
    VERSION = 11
    TYPE = "thermal"

    @property
    def tracker_version(self):
        return self.version

    @property
    def type(self):
        return ClipTrackExtractor.TYPE

    def __init__(
        self,
        config,
    ):
        self.version = f"PI-{ClipTrackExtractor.VERSION}"
        self.background_alg = WeightedBackground()

        self.update_background = True
        self.calculate_filtered = True
        self.weighting_percent = 1

        self.calculate_thumbnail_info = False
        self.do_tracking = True
        self.config = config
        self.stats = None
        self.max_tracks = config.max_tracks
        self.frame_padding = max(3, self.config.frame_padding)
        self.keep_frames = True
        self.calc_stats = False
        self._tracking_time = None
        self.min_dimension = config.min_dimension
        self.background_thresh = 50

    @property
    def tracking_time(self):
        return self._tracking_time

    def _get_filtered_frame(self, thermal, sub_change=True, denoise=False):
        """
        Calculates filtered frame from thermal
        :param thermal: the thermal frame
        :param background: (optional) used for background subtraction
        :return: uint8 filtered frame and adjusted clip threshold for normalized frame
        """
        from tools import normalize

        filtered = np.float32(thermal.copy())
        if sub_change:
            avg_change = int(
                round(np.average(thermal) - self.background_alg.get_average())
            )
        else:
            avg_change = 0

        np.clip(
            filtered - self.background_alg.background - avg_change,
            0,
            None,
            out=filtered,
        )
        filtered, stats = normalize(filtered, new_max=255)
        if denoise:
            from cv2 import fastNlMeansDenoising

            filtered = fastNlMeansDenoising(np.uint8(filtered), None)
        if stats[1] == stats[2]:
            mapped_thresh = self.background_thresh
        else:
            mapped_thresh = self.background_thresh / (stats[1] - stats[2]) * 255
        return filtered, mapped_thresh

    def process_frame(self, clip, frame, **args):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
        If specified background subtraction algorithm will be used.
        """
        from tools import is_affected_by_ffc

        ffc_affected = is_affected_by_ffc(frame)

        self.background_alg.process_frame(frame, ffc_affected)

        thermal = frame.pix.copy()
        clip.ffc_affected = ffc_affected
        mask = None
        filtered = None
        if self.do_tracking or self.calculate_filtered or self.calculate_thumbnail_info:
            filtered = np.float32(frame.pix) - self.background_alg.background
        if self.do_tracking or self.calculate_thumbnail_info:
            from tools import detect_objects

            obj_filtered, threshold = self._get_filtered_frame(
                thermal, denoise=self.config.denoise
            )

            _, mask, component_details, centroids = detect_objects(
                obj_filtered, otsus=False, threshold=threshold, kernel=(5, 5)
            )
        _ = clip.add_frame(thermal, filtered, mask, ffc_affected)

        regions = []
        if ffc_affected:
            clip.active_tracks = set()
            stale_tracks = []
        else:
            regions = self._get_regions_of_interest(
                clip, component_details[1:], centroids[1:]
            )
            stale_tracks = self._apply_region_matchings(clip, regions)
        clip.region_history.append(regions)
        return stale_tracks

    def get_delta_frame(self, clip):
        from tools import normalize

        frame = clip.frame_buffer.current_frame
        prev_frame = clip.frame_buffer.prev_frame
        if prev_frame is None:
            return None, None
        filtered, _ = normalize(frame.filtered, new_max=255)
        prev_filtered, _ = normalize(prev_frame.filtered, new_max=255)
        delta_filtered = np.abs(np.float32(filtered) - np.float32(prev_filtered))
        return delta_filtered

    def _get_regions_of_interest(self, clip, component_details, centroids=None):
        """
        Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
        rectangles.
        :param filtered: The filtered frame
        :return: regions of interest, mask frame
        """
        from region import Region
        import math

        delta_filtered = self.get_delta_frame(
            clip,
        )

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = self.frame_padding
        # find regions of interest
        regions = []
        for i, component in enumerate(component_details):
            if centroids is None:
                centroid = [
                    int(component[0] + component[2] / 2),
                    int(component[1] + component[3] / 2),
                ]
            else:
                centroid = centroids[i]
            region = Region(
                component[0],
                component[1],
                component[2],
                component[3],
                mass=component[4],
                id=i,
                frame_number=clip.current_frame,
                centroid=centroid,
            )

            # Make this min area
            if region.width < self.min_dimension or region.height < self.min_dimension:
                continue
            # GP this needs to be checked for themals 29/06/2022
            if delta_filtered is not None:
                region_difference = region.subimage(delta_filtered)
                region.pixel_variance = np.var(region_difference)
            old_region = region.copy()
            region.crop(clip.crop_rectangle)
            region.was_cropped = str(old_region) != str(region)

            if self.config.cropped_regions_strategy == "cautious":
                crop_width_fraction = (
                    old_region.width - region.width
                ) / old_region.width
                crop_height_fraction = (
                    old_region.height - region.height
                ) / old_region.height
                if crop_width_fraction > 0.25 or crop_height_fraction > 0.25:
                    continue
            elif (
                self.config.cropped_regions_strategy == "none"
                or self.config.cropped_regions_strategy is None
            ):
                if region.was_cropped:
                    continue
            elif self.config.cropped_regions_strategy != "all":
                raise ValueError(
                    "Invalid mode for CROPPED_REGIONS_STRATEGY, expected ['all','cautious','none'] but found {}".format(
                        self.config.cropped_regions_strategy
                    )
                )

            # filter out regions that are probably just noise
            if self.config.filter_regions_pre_match and (
                region.pixel_variance < self.config.aoi_pixel_variance
                and region.mass < self.config.aoi_min_mass
            ):
                logging.debug(
                    "%s filtering region %s because of variance %s and mass %s",
                    region.frame_number,
                    region,
                    region.pixel_variance,
                    region.mass,
                )
                continue

            region.enlarge(padding, max=clip.crop_rectangle)
            # gp dunno if we should use this feels like we already have the edge
            # extra_edge = 0
            extra_edge = math.ceil(clip.crop_rectangle.width * 0.03)
            region.set_is_along_border(clip.crop_rectangle, edge=extra_edge)
            regions.append(region)
        return regions

    def _apply_region_matchings(self, clip, regions):
        """
        Work out the best matchings between tracks and regions of interest for the current frame.
        Create any new tracks required.
        """
        unmatched_regions, matched_tracks = self._match_existing_tracks(clip, regions)
        new_tracks = self._create_new_tracks(clip, unmatched_regions)

        unactive_tracks = clip.active_tracks - matched_tracks - new_tracks
        clip.active_tracks = matched_tracks | new_tracks
        return self._filter_inactive_tracks(clip, unactive_tracks)

    def _match_existing_tracks(self, clip, regions):

        scores = []
        used_regions = set()
        unmatched_regions = set(regions)
        active = list(clip.active_tracks)
        active.sort(key=lambda x: x.get_id())
        for track in active:
            scores.extend(track.match(regions))

        # makes tracking consistent by ordering by score then by frame since target then track id
        scores.sort(
            key=lambda record: record[1].frames_since_target_seen
            + float(".{}".format(record[1]._id))
        )
        scores.sort(key=lambda record: record[0])
        matched_tracks = set()
        blanked_tracks = set()
        cur_frame = clip.frame_buffer.current_frame
        for score, track, region in scores:
            if (
                track in matched_tracks
                or region in used_regions
                or track in blanked_tracks
            ):
                continue
            logging.debug(
                "frame# %s matched %s to track %s", clip.current_frame, region, track
            )
            used_regions.add(region)
            unmatched_regions.remove(region)
            if not self.config.filter_regions_pre_match:
                if self.config.min_hist_diff is not None:
                    from ml_tools.imageprocessing import hist_diff

                    background = self.background_alg.background
                    # if self.scale:
                    #     background = clip.rescaled_background(
                    #         (int(self.res_x), int(self.res_y))
                    #     )
                    hist_v = hist_diff(region, background, cur_frame.thermal)
                    if hist_v > self.config.min_hist_diff:
                        logging.warn(
                            "%s filtering region %s because of hist diff %s track %s ",
                            region.frame_number,
                            region,
                            hist_v,
                            track,
                        )

                        blanked_tracks.add(track)
                        continue
                if (
                    region.pixel_variance < self.config.aoi_pixel_variance
                    or region.mass < self.config.aoi_min_mass
                ):
                    # this will force a blank frame to be added, rather than if we filter earlier
                    # and match this track to a different region
                    logging.debug(
                        "%s filtering region %s because of variance %s and mass %s track %s",
                        region.frame_number,
                        region,
                        region.pixel_variance,
                        region.mass,
                        track,
                    )
                    blanked_tracks.add(track)
                    continue
            track.add_region(region)
            matched_tracks.add(track)

        return unmatched_regions, matched_tracks

    def _create_new_tracks(self, clip, unmatched_regions):
        """Create new tracks for any unmatched regions"""
        from track import Track

        new_tracks = set()
        for region in unmatched_regions:
            # make sure we don't overlap with existing tracks.  This can happen if a tail gets tracked as a new object
            overlaps = [
                track.last_bound.overlap_area(region) for track in clip.active_tracks
            ]
            if len(overlaps) > 0 and max(overlaps) > (region.area * 0.25):
                continue

            track = Track.from_region(
                clip,
                region,
                self.tracker_version,
                tracking_config=self.config,
            )
            new_tracks.add(track)
            clip._add_active_track(track)

        return new_tracks

    def _filter_inactive_tracks(self, clip, unactive_tracks):
        """Filters tracks which are or have become inactive"""
        stale_tracks = []
        for track in unactive_tracks:
            track.add_blank_frame()
            if track.tracking:
                clip.active_tracks.add(track)
                logging.debug(
                    "frame {} adding a blank frame to {} ".format(
                        clip.current_frame, track.get_id()
                    )
                )
            else:
                stale_tracks.append(track)
        return stale_tracks


def is_affected_by_ffc(cptv_frame):
    from datetime import timedelta

    if hasattr(cptv_frame, "ffc_status") and cptv_frame.ffc_status in [1, 2]:
        return True

    if cptv_frame.time_on is None or cptv_frame.last_ffc_time is None:
        return False
    if isinstance(cptv_frame.time_on, int):
        return (cptv_frame.time_on - cptv_frame.last_ffc_time) < timedelta(
            seconds=9.9
        ).seconds
    return (cptv_frame.time_on - cptv_frame.last_ffc_time) < timedelta(seconds=9.9)
