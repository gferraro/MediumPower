# cython: language_level=3
# cython: boundscheck=False, wraparound=False
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

import logging
import math
import numpy as np
cimport numpy as cnp
from collections import namedtuple

from tracker import Tracker

class RegionTracker(Tracker):
    MIN_KALMAN_FRAMES = 18

    MASS_CHANGE_PERCENT = 0.55

    # IR VALUES
    BASE_DISTANCE_CHANGE = 11250

    MIN_MASS_CHANGE = 20 * 4
    RESTRICT_MASS_AFTER = 1.5

    MAX_DISTANCE = 30752
    BASE_VELOCITY = 8
    VELOCITY_MULTIPLIER = 10

    def __init__(self, int id, tracking_config, crop_rectangle=None):
        self.track_id = id
        self.clear_run = 0
        from kalman import Kalman

        self.kalman_tracker = Kalman()
        self._frames_since_target_seen = 0
        self.frames = 0
        self._blank_frames = 0
        self._last_bound = None
        self.crop_rectangle = crop_rectangle
        self._tracking = False
        self.type = tracking_config.type
        self.min_mass_change = tracking_config.params.get(
            "min_mass_change", RegionTracker.MIN_MASS_CHANGE
        )
        self.max_distance = tracking_config.params.get(
            "max_distance", RegionTracker.MAX_DISTANCE
        )
        self.base_distance_change = tracking_config.params.get(
            "base_distance_change", RegionTracker.BASE_DISTANCE_CHANGE
        )
        self.restrict_mass_after = tracking_config.params.get(
            "restrict_mass_after", RegionTracker.RESTRICT_MASS_AFTER
        )
        self.mass_change_percent = tracking_config.params.get(
            "mass_change_percent", RegionTracker.MASS_CHANGE_PERCENT
        )
        self.velocity_multiplier = tracking_config.params.get(
            "velocity_multiplier", RegionTracker.VELOCITY_MULTIPLIER
        )
        self.base_velocity = tracking_config.params.get(
            "base_velocity", RegionTracker.BASE_VELOCITY
        )
        self.max_blanks = tracking_config.params.get("max_blanks", 18)

    @property
    def tracking(self):
        return self._tracking

    @property
    def last_bound(self):
        return self._last_bound

    def get_size_change(self, double current_area, region):
        """Gets a value representing the difference in regions sizes"""
        cdef double size_difference = abs(region.area - current_area) / (current_area + 50)
        return size_difference

    def match(self, regions, track):
        cdef double distance_score, avg_mass_val
        scores = []
        avg_mass = track.average_mass()
        max_distances = self.get_max_distance_change(track)
        for region in regions:
            size_change = self.get_size_change(track.average_area(), region)
            distances = self.last_bound.average_distance(region)

            max_size_change = get_max_size_change(track, region)
            max_mass_change = self.get_max_mass_change_percent(track, avg_mass)

            logging.debug(
                "Track %s %s has max size change %s, distances %s to region %s size change %s max distance %s",
                track,
                track.last_bound,
                max_size_change,
                distances,
                region,
                size_change,
                max_distances,
            )
            if type == "thermal":
                distances = [np.mean(distances)]
                max_distances = max_distances[:1]
            else:
                distances = [(distances[0] + distances[2]) / 2]
                max_distances = max_distances[:1]

            if max_mass_change and abs(avg_mass - region.mass) > max_mass_change:
                logging.debug(
                    "track %s region mass %s deviates too much from %s for region %s",
                    track.get_id(),
                    region.mass,
                    avg_mass,
                    region,
                )
                continue
            skip = False
            for distance, max_distance in zip(distances, max_distances):
                if max_distance is None:
                    continue
                if distance > max_distance:
                    logging.debug(
                        "track %s distance score %s bigger than max distance %s for region %s",
                        track.get_id(),
                        distance,
                        max_distance,
                        region,
                    )
                    skip = True
                    break
            if skip:
                continue

            if size_change > max_size_change:
                logging.debug(
                    "track % size_change %s bigger than max size_change %s for region %s",
                    track.get_id(),
                    size_change,
                    max_size_change,
                    region,
                )
                continue
            if type == "ir":
                distance_score = np.mean(distances)
            else:
                distance_score = distances[0]

            scores.append((distance_score, track, region))
        return scores

    def add_region(self, region):
        cdef int stop_tracking
        self.frames += 1
        if region.blank:
            self._blank_frames += 1
            self._frames_since_target_seen += 1
            stop_tracking = min(
                2 * (self.frames - self._frames_since_target_seen),
                self.max_blanks,
            )
            self._tracking = self._frames_since_target_seen < stop_tracking
        else:
            if self._frames_since_target_seen != 0:
                self.clear_run = 0
            self.clear_run += 1
            self._tracking = True
            self.kalman_tracker.correct(region)
            self._frames_since_target_seen = 0

        prediction = self.kalman_tracker.predict()
        self.predicted_mid = (prediction[0][0], prediction[1][0])
        self._last_bound = region

    @property
    def blank_frames(self):
        return self._blank_frames

    @property
    def frames_since_target_seen(self):
        return self._frames_since_target_seen

    @property
    def nonblank_frames(self):
        return self.frames - self._blank_frames

    def predicted_velocity(self):
        cdef double pred_vel_x, pred_vel_y
        if (
            self.last_bound is None
            or self.nonblank_frames <= RegionTracker.MIN_KALMAN_FRAMES
        ):
            return (0, 0)
        pred_vel_x = self.predicted_mid[0] - self.last_bound.centroid[0]
        pred_vel_y = self.predicted_mid[1] - self.last_bound.centroid[1]
        return (pred_vel_x, pred_vel_y)

    def add_blank_frame(self):
        from region import Region

        cdef int kalman_amount = (
            self.frames
            - RegionTracker.MIN_KALMAN_FRAMES
            - self._frames_since_target_seen * 2
        )

        if kalman_amount > 0:
            region = Region(
                int(self.predicted_mid[0] - self.last_bound.width / 2.0),
                int(self.predicted_mid[1] - self.last_bound.height / 2.0),
                self.last_bound.width,
                self.last_bound.height,
                centroid=[self.predicted_mid[0], self.predicted_mid[1]],
            )
            if self.crop_rectangle:
                region.crop(self.crop_rectangle)
        else:
            region = self.last_bound.copy()
        region.blank = True
        region.mass = 0
        region.pixel_variance = 0
        region.frame_number = self.last_bound.frame_number + 1

        self.add_region(region)
        return region

    def tracker_version(self):
        return f"RegionTracker-{RegionTracker.TRACKER_VERSION}"

    def get_max_distance_change(self, track):
        cdef double x, y, velocity_distance, pred_distance, max_distance
        x, y = track.velocity
        if len(track) == 1:
            x = self.base_velocity
            y = self.base_velocity
        x = self.velocity_multiplier * x
        y = self.velocity_multiplier * y
        velocity_distance = x * x + y * y

        pred_vel = track.predicted_velocity()
        logging.debug(
            "%s velo %s pred vel %s vel distance %s",
            track,
            track.velocity,
            track.predicted_velocity(),
            velocity_distance,
        )
        pred_distance = pred_vel[0] * pred_vel[0] + pred_vel[1] * pred_vel[1]
        pred_distance = max(velocity_distance, pred_distance)
        max_distance = self.base_distance_change + max(velocity_distance, pred_distance)
        distances = [max_distance, None, max_distance]
        return distances

    def get_max_mass_change_percent(self, track, double average_mass):
        cdef double mass_percent
        if self.mass_change_percent is None:
            return None
        if len(track) > self.restrict_mass_after * track.fps:
            vel = track.velocity
            mass_percent = self.mass_change_percent
            if np.sum(np.abs(vel)) > 5:
                mass_percent = mass_percent + 0.1
            return max(
                self.min_mass_change,
                average_mass * mass_percent,
            )
        else:
            return None


def get_max_size_change(track, region):
    cdef double region_percent, vel
    cdef bint exiting, entering
    exiting = region.is_along_border and not track.last_bound.is_along_border
    entering = not exiting and track.last_bound.is_along_border
    region_percent = 1.5
    if len(track) < 5:
        region_percent = 2.0
    vel = np.sum(np.abs(track.velocity))
    if entering or exiting:
        region_percent = 2.0
        if vel > 10:
            region_percent *= 3
    elif vel > 10:
        region_percent *= 2
    return region_percent


class ThumbInfo:
    def __init__(self, track_id):
        self.points = -1
        self.region = None
        self.thumb = None
        self.thumb_frame = None
        self.last_frame_check = None
        self.predicted_tag = None
        self.predicted_confidence = None
        self.track_id = track_id

    def score(self):
        cdef int confidence_threshold = 80
        cdef double score = self.points
        cdef double score_offset = 100000
        cdef double confidence

        if self.predicted_tag is not None:
            if self.predicted_tag != "false-positive":
                score = score + 1000 * score_offset
                if self.predicted_confidence > confidence_threshold:
                    confidence = self.predicted_confidence
                else:
                    confidence = 0
            else:
                confidence = 100 - self.predicted_confidence

            score = score + confidence * score_offset
        return score

    def to_metadata(self):
        thumbnail_info = {
            "region": self.region.meta_dictionary(),
            "contours": self.points,
            "score": round(self.score()),
        }
        return thumbnail_info


class Track:
    """Bounds of a tracked object over time."""

    _track_id = 1

    JITTER_THRESHOLD = 0.3
    MIN_JITTER_CHANGE = 5

    def __init__(
        self,
        clip_id,
        id=None,
        fps=9,
        tracking_config=None,
        crop_rectangle=None,
        tracker_version=None,
    ):
        self.in_trap = False
        self.trap_reported = False
        self.trigger_frame = None
        self.direction = 0
        self.trap_tag = None
        if not id:
            self._id = Track._track_id
            Track._track_id += 1
        else:
            self._id = id
        self.clip_id = clip_id
        self.start_frame = None
        self.start_s = None
        self.end_s = None
        self.fps = fps
        self.current_frame_num = None
        self.frame_list = []
        self.bounds_history = []

        self.vel_x = []
        self.vel_y = []
        self.tag = "unknown"
        self.prev_frame_num = None
        self.confidence = None
        self.max_novelty = None
        self.avg_novelty = None

        self.from_metadata = False
        self.tags = None
        self.predictions = None
        self.predicted_class = None
        self.predicted_confidence = None

        self.all_class_confidences = None
        self.prediction_classes = None

        self.crop_rectangle = crop_rectangle

        self.predictions = None
        self.predicted_tag = None
        self.predicted_confidence = None

        self.all_class_confidences = None
        self.prediction_classes = None
        self.tracker_version = tracker_version

        self.tracker = None
        if tracking_config is not None:
            self.tracker = self.get_tracker(tracking_config)
        self.thumb_info = None
        self.score = None

    def get_tracker(self, tracking_config):
        tracker = tracking_config.tracker
        if tracker == "RegionTracker":
            return RegionTracker(self.get_id(), tracking_config, self.crop_rectangle)
        else:
            raise Exception(f"Cant find for tracker {tracker}")

    @property
    def blank_frames(self):
        if self.tracker is None:
            return 0
        return self.tracker.blank_frames

    @property
    def tracking(self):
        return self.tracker.tracking

    @property
    def frames_since_target_seen(self):
        return self.tracker.frames_since_target_seen

    def match(self, regions):
        return self.tracker.match(regions, self)

    def get_segments(
        self,
        segment_width,
        segment_frame_spacing=9,
        repeats=1,
        min_frames=0,
        segment_frames=None,
        segment_types=None,
        from_last=None,
        max_segments=None,
        ffc_frames=None,
        dont_filter=False,
        filter_by_fp=False,
        min_segments=1,
    ):
        from ml_tools.datasetstructures import get_segments, SegmentHeader

        if from_last is not None:
            if from_last == 0:
                return []
            regions = np.array(self.bounds_history[-from_last:])
            start_frame = regions[0].frame_number
        else:
            start_frame = self.start_frame
            regions = np.array(self.bounds_history)

        segments = []
        if segment_frames is not None:
            mass_history = np.uint16([region.mass for region in regions])
            for frames in segment_frames:
                relative_frames = frames - self.start_frame
                mass_slice = mass_history[relative_frames]
                segment_mass = np.sum(mass_slice)
                segment = SegmentHeader(
                    self.clip_id,
                    self._id,
                    start_frame=start_frame,
                    frames=len(frames),
                    weight=1,
                    mass=segment_mass,
                    label=None,
                    regions=regions[relative_frames],
                    frame_indices=frames,
                )
                segments.append(segment)
        else:
            segments, _ = get_segments(
                self.clip_id,
                self._id,
                start_frame,
                segment_frame_spacing=segment_frame_spacing,
                segment_width=segment_width,
                regions=regions,
                ffc_frames=ffc_frames,
                repeats=repeats,
                min_frames=min_frames,
                segment_types=segment_types,
                max_segments=max_segments,
                dont_filter=dont_filter,
                min_segments=min_segments,
            )

        return segments

    @classmethod
    def from_region(cls, clip, region, tracker_version=None, tracking_config=None):
        track = cls(
            clip.get_id(),
            fps=clip.frames_per_second,
            tracker_version=tracker_version,
            crop_rectangle=clip.crop_rectangle,
            tracking_config=tracking_config,
        )
        track.start_frame = region.frame_number
        track.start_s = region.frame_number / float(clip.frames_per_second)
        track.add_region(region)
        return track

    def get_id(self):
        return self._id

    def add_prediction_info(self, track_prediction):
        logging.warn("TODO add prediction info needs to be implemented")
        return


    def add_region(self, region):
        cdef int frame_diff
        if self.prev_frame_num and region.frame_number:
            frame_diff = region.frame_number - self.prev_frame_num - 1
            for _ in range(frame_diff):
                self.add_blank_frame()
        self.tracker.add_region(region)
        self.bounds_history.append(region)
        self.prev_frame_num = region.frame_number
        self.update_velocity()

    def update_velocity(self):
        if len(self.bounds_history) >= 2:
            self.vel_x.append(
                self.bounds_history[-1].centroid[0]
                - self.bounds_history[-2].centroid[0]
            )
            self.vel_y.append(
                self.bounds_history[-1].centroid[1]
                - self.bounds_history[-2].centroid[1]
            )
        else:
            self.vel_x.append(0)
            self.vel_y.append(0)

    def crop_regions(self):
        if self.crop_rectangle is None:
            logging.info("No crop rectangle to crop with")
            return
        for region in self.bounds_history:
            region.crop(self.crop_rectangle)

    def add_frame_for_existing_region(self, frame, mass_delta_threshold, prev_filtered):
        logging.error("add_frame_for_existing_region is not implemented anymore")

    def average_area(self):
        """Average area of last 5 frames that weren't blank"""
        cdef double avg_area = 0.0
        cdef int count = 0, i
        for i in range(len(self.bounds_history)):
            bound = self.bounds_history[-i - 1]
            if not bound.blank:
                avg_area += bound.area
                count += 1
            if count == 5:
                break
        if count == 0:
            return 0
        return avg_area / count

    def average_mass(self):
        """Average mass of last 5 frames that weren't blank"""
        cdef double avg_mass = 0.0
        cdef int count = 0, i
        for i in range(len(self.bounds_history)):
            bound = self.bounds_history[-i - 1]
            if not bound.blank:
                avg_mass += bound.mass
                count += 1
            if count == 5:
                break
        if count == 0:
            return 0
        return avg_mass / count

    def add_blank_frame(self):
        """Maintains same bounds as previously, does not reset frames_since_target_seen counter"""
        if self.tracker:
            region = self.tracker.add_blank_frame()
        self.bounds_history.append(region)
        self.prev_frame_num = region.frame_number
        self.update_velocity()

    def calculate_stats(self):
        from tools import eucl_distance_sq

        cdef double movement = 0.0, max_offset = 0.0, avg_vel = 0.0
        cdef double distance, offset
        cdef int frames_moved = 0, jitter_bigger = 0, jitter_smaller = 0
        cdef int jitter_percent, blank_percent
        cdef double vx, vy, height_diff, width_diff, thresh_h, thresh_v
        cdef double movement_points, delta_points, score

        if len(self) <= 1:
            self.stats = TrackMovementStatistics()
            return
        non_blank = [bound for bound in self.bounds_history if not bound.blank]
        mass_history = [int(bound.mass) for bound in non_blank]
        variance_history = [
            bound.pixel_variance for bound in non_blank if bound.pixel_variance
        ]

        first_point = self.bounds_history[0].mid
        for i, (vx, vy) in enumerate(zip(self.vel_x, self.vel_y)):
            region = self.bounds_history[i]
            if not region.blank:
                avg_vel += abs(vx) + abs(vy)
            if i == 0:
                continue

            if region.blank or self.bounds_history[i - 1].blank:
                continue
            if region.has_moved(self.bounds_history[i - 1]) or region.is_along_border:
                distance = (vx * vx + vy * vy) ** 0.5
                movement += distance
                offset = eucl_distance_sq(first_point, region.mid)
                if offset > max_offset:
                    max_offset = offset
                frames_moved += 1
        avg_vel = avg_vel / len(mass_history)
        max_offset = math.sqrt(max_offset)
        delta_std = float(np.mean(variance_history)) ** 0.5
        for i, bound in enumerate(self.bounds_history[1:]):
            prev_bound = self.bounds_history[i]
            if prev_bound.is_along_border or bound.is_along_border:
                continue
            height_diff = bound.height - prev_bound.height
            width_diff = prev_bound.width - bound.width
            thresh_h = max(
                Track.MIN_JITTER_CHANGE, prev_bound.height * Track.JITTER_THRESHOLD
            )
            thresh_v = max(
                Track.MIN_JITTER_CHANGE, prev_bound.width * Track.JITTER_THRESHOLD
            )
            if abs(height_diff) > thresh_h:
                if height_diff > 0:
                    jitter_bigger += 1
                else:
                    jitter_smaller += 1
            elif abs(width_diff) > thresh_v:
                if width_diff > 0:
                    jitter_bigger += 1
                else:
                    jitter_smaller += 1

        movement_points = (movement ** 0.5) + max_offset
        delta_points = delta_std * 25.0
        jitter_percent = int(
            round(100 * (jitter_bigger + jitter_smaller) / float(self.frames))
        )

        blank_percent = int(round(100.0 * self.blank_frames / self.frames))
        score = (
            min(movement_points, 100)
            + min(delta_points, 100)
            + (100 - jitter_percent)
            + (100 - blank_percent)
        )
        stats = TrackMovementStatistics(
            movement=float(movement),
            max_offset=float(max_offset),
            average_mass=float(np.mean(mass_history)),
            median_mass=float(np.median(mass_history)),
            delta_std=float(delta_std),
            score=float(score),
            region_jitter=jitter_percent,
            jitter_bigger=jitter_bigger,
            jitter_smaller=jitter_smaller,
            blank_percent=blank_percent,
            frames_moved=frames_moved,
            mass_std=float(np.std(mass_history)),
            average_velocity=float(avg_vel),
        )
        self.stats = stats

    def smooth(self, frame_bounds):
        """Smooths out any quick changes in track dimensions"""
        cdef double frame_x, frame_y, frame_width, frame_height
        if len(self.bounds_history) == 0:
            return

        from region import Region

        new_bounds_history = []
        for i in range(len(self.bounds_history)):
            prev_frame = self.bounds_history[max(0, i - 1)]
            current_frame = self.bounds_history[i]
            next_frame = self.bounds_history[min(len(self.bounds_history) - 1, i + 1)]

            frame_x = current_frame.centroid[0]
            frame_y = current_frame.centroid[1]
            frame_width = (prev_frame.width + current_frame.width + next_frame.width) / 3
            frame_height = (prev_frame.height + current_frame.height + next_frame.height) / 3
            frame = Region(
                int(frame_x - frame_width / 2),
                int(frame_y - frame_height / 2),
                int(frame_width),
                int(frame_height),
            )
            frame.crop(frame_bounds)
            new_bounds_history.append(frame)

        self.bounds_history = new_bounds_history

    def trim(self):
        """Removes empty frames from start and end of track"""
        cdef int start = 0, end
        mass_history = [int(bound.mass) for bound in self.bounds_history]
        median_mass = np.median(mass_history)
        cdef double filter_mass = 0.005 * median_mass
        filter_mass = max(filter_mass, 2)
        logging.debug(
            "Triming track with median % and filter mass %s", median_mass, filter_mass
        )
        while start < len(self) and mass_history[start] <= filter_mass:
            start += 1
        end = len(self) - 1

        while end > 0 and mass_history[end] <= filter_mass:
            if self.tracker and self.frames_since_target_seen > 0:
                self.tracker._frames_since_target_seen -= 1
                self.tracker._blank_frames -= 1
            end -= 1
        if end < start:
            self.bounds_history = []
            self.vel_x = []
            self.vel_y = []
            if self.tracker:
                self.tracker._blank_frames = 0
        else:
            self.start_frame += start
            self.bounds_history = self.bounds_history[start:end + 1]
            self.vel_x = self.vel_x[start:end + 1]
            self.vel_y = self.vel_y[start:end + 1]
        self.start_s = self.start_frame / float(self.fps)

    def get_overlap_ratio(self, other_track, double threshold=0.05):
        """Checks what ratio of the time these two tracks overlap."""
        cdef int start, end, pos, our_index, other_index, frames_overlapped = 0
        cdef double overlap

        if len(self) == 0 or len(other_track) == 0:
            return 0.0

        start = max(self.start_frame, other_track.start_frame)
        end = min(self.end_frame, other_track.end_frame)

        for pos in range(start, end + 1):
            our_index = pos - self.start_frame
            other_index = pos - other_track.start_frame
            if (
                our_index >= 0
                and other_index >= 0
                and our_index < len(self)
                and other_index < len(other_track)
            ):
                our_bounds = self.bounds_history[our_index]
                if our_bounds.area == 0:
                    continue
                other_bounds = other_track.bounds_history[other_index]
                overlap = our_bounds.overlap_area(other_bounds) / our_bounds.area
                if overlap >= threshold:
                    frames_overlapped += 1

        return frames_overlapped / len(self)

    def set_end_s(self, fps):
        if len(self) == 0:
            self.end_s = self.start_s
            return
        self.end_s = (self.end_frame + 1) / fps

    def predicted_velocity(self):
        return self.tracker.predicted_velocity()

    def update_trapped_state(self):
        if self.in_trap:
            return self.in_trap
        cdef int min_frames = 2
        if len(self.bounds_history) < min_frames:
            return False
        self.in_trap = all(r.in_trap for r in self.bounds_history[-min_frames:])
        return self.in_trap

    @property
    def end_frame(self):
        if len(self.bounds_history) == 0:
            return self.start_frame
        return self.bounds_history[-1].frame_number

    @property
    def nonblank_frames(self):
        return self.end_frame + 1 - self.start_frame - self.blank_frames

    @property
    def frames(self):
        return self.end_frame + 1 - self.start_frame

    @property
    def last_mass(self):
        return self.bounds_history[-1].mass

    @property
    def velocity(self):
        return self.vel_x[-1], self.vel_y[-1]

    @property
    def last_bound(self):
        return self.bounds_history[-1]

    def __repr__(self):
        return "Track: {} frames# {}".format(self.get_id(), len(self))

    def __len__(self):
        return len(self.bounds_history)

    def start_and_end_in_secs(self):
        if self.end_s is None:
            if len(self) == 0:
                self.end_s = self.start_s
            else:
                self.end_s = (self.end_frame + 1) / self.fps

        return (self.start_s, self.end_s)

    def get_metadata(self, predictions_per_model=None):
        track_info = {}
        start_s, end_s = self.start_and_end_in_secs()

        track_info["id"] = self.get_id()
        if self.in_trap:
            track_info["trap_triggered"] = self.in_trap
            track_info["trigger_frame"] = self.trigger_frame
            if self.trap_tag is not None:
                track_info["trap_tag"] = self.trap_tag
        track_info["tracker_version"] = self.tracker_version
        track_info["start_s"] = round(start_s, 2)
        track_info["end_s"] = round(end_s, 2)
        track_info["num_frames"] = len(self)
        track_info["frame_start"] = self.start_frame
        track_info["frame_end"] = self.end_frame
        track_info["positions"] = self.bounds_history
        if self.thumb_info is not None:
            track_info["thumbnail"] = self.thumb_info.to_metadata()
        track_info["tracking_score"] = 0 if self.stats is None else self.stats.score
        prediction_info = []
        if predictions_per_model:
            for model_id, predictions in predictions_per_model.items():
                prediction = predictions.prediction_for(self.get_id())
                if prediction is None:
                    continue
                prediciont_meta = prediction.get_metadata(predictions.thresholds)
                prediciont_meta["model_id"] = model_id
                prediction_info.append(prediciont_meta)
        track_info["predictions"] = prediction_info
        return track_info

    @classmethod
    def get_best_human_tag(cls, track_tags, tag_precedence, min_confidence=-1):
        """returns highest precedence non AI tag from the metadata"""
        if track_tags is None:
            return None
        track_tags = [
            tag
            for tag in track_tags
            if not tag.get("automatic", False)
            and tag.get("confidence") >= min_confidence
        ]

        if not track_tags:
            return None

        tag = None
        if tag_precedence is None:
            default_prec = 100
            tag_precedence = {}
        else:
            default_prec = tag_precedence.get("default", 100)
        best = None
        for track_tag in track_tags:
            ranking = cls.tag_ranking(track_tag, tag_precedence, default_prec)
            if tag and ranking == best:
                if is_conflicting_tag(tag, track_tag):
                    tag = None
                else:
                    path_one = tag.get("path")
                    path_two = track_tag.get("path")
                    if len(path_two) > len(path_one):
                        tag = track_tag
            elif best is None or ranking < best:
                best = ranking
                tag = track_tag
        return tag

    @staticmethod
    def tag_ranking(track_tag, precedence, default_prec):
        """returns a ranking of tags based of what they are and confidence"""
        cdef double confidence, prec
        what = track_tag.get("what")
        confidence = 1 - track_tag.get("confidence", 0)
        prec = precedence.get(what, default_prec)
        return prec + confidence


def is_conflicting_tag(tag_one, tag_two):
    path_one = tag_one.get("path")
    path_two = tag_two.get("path")
    same_parents = path_one in path_two or path_two in path_one
    same_parents and path_one != "all.mammal" and path_two != "all.mammal"
    return tag_one["what"] != tag_two["what"] and not same_parents


TrackMovementStatistics = namedtuple(
    "TrackMovementStatistics",
    "movement max_offset score average_mass median_mass delta_std region_jitter jitter_smaller jitter_bigger blank_percent frames_moved mass_std, average_velocity",
)
TrackMovementStatistics.__new__.__defaults__ = (0,) * len(
    TrackMovementStatistics._fields
)
