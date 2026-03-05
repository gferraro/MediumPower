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
import numpy as np

import logging
from tracker import Tracker


class RegionTracker(Tracker):
    # number of frames required before using kalman estimation
    MIN_KALMAN_FRAMES = 18

    # THERMAL VALUES
    # GP Need to put in config per camera type
    # MAX_DISTANCE = 2000
    #
    # TRACKER_VERSION = 1
    # BASE_DISTANCE_CHANGE = 450
    # # minimum region mass change
    # MIN_MASS_CHANGE = 20
    # # enforce mass growth after X seconds
    # RESTRICT_MASS_AFTER = 1.5
    # # amount region mass can change
    MASS_CHANGE_PERCENT = 0.55

    # IR VALUES
    BASE_DISTANCE_CHANGE = 11250

    # minimum region mass change
    MIN_MASS_CHANGE = 20 * 4
    # enforce mass growth after X seconds
    RESTRICT_MASS_AFTER = 1.5
    # amount region mass can change

    # MAX_DISTANCE = 2000
    MAX_DISTANCE = 30752
    BASE_VELOCITY = 8
    VELOCITY_MULTIPLIER = 10

    def __init__(self, id, tracking_config, crop_rectangle=None):
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

    def get_size_change(self, current_area, region):
        """
        Gets a value representing the difference in regions sizes
        """

        # ratio of 1.0 = 20 points, ratio of 2.0 = 10 points, ratio of 3.0 = 0 points.
        # area is padded with 50 pixels so small regions don't change too much
        size_difference = abs(region.area - current_area) / (current_area + 50)

        return size_difference

    def match(self, regions, track):
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
            # only for thermal
            if type == "thermal":
                # GP should figure out good values for the 3 distances rather than the mean
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
                    # continue
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
            # only for thermal
            if type == "ir":
                distance_score = np.mean(distances)
            else:
                # GP should figure out good values for the 3 distances rather than the mean
                distance_score = distances[0]

            scores.append((distance_score, track, region))
        return scores

    def add_region(self, region):
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

        kalman_amount = (
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
        x, y = track.velocity
        # x = max(x, 2)
        # y = max(y, 2)
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
        # max top left, max between predicted and region, max between right bottom
        distances = [max_distance, None, max_distance]
        return distances

    def get_max_mass_change_percent(self, track, average_mass):
        if self.mass_change_percent is None:
            return None
        if len(track) > self.restrict_mass_after * track.fps:
            vel = track.velocity
            mass_percent = self.mass_change_percent
            if np.sum(np.abs(vel)) > 5:
                # faster tracks can be a bit more deviant
                mass_percent = mass_percent + 0.1
            return max(
                self.min_mass_change,
                average_mass * mass_percent,
            )
        else:
            return None


def get_max_size_change(track, region):
    exiting = region.is_along_border and not track.last_bound.is_along_border
    entering = not exiting and track.last_bound.is_along_border
    region_percent = 1.5
    if len(track) < 5:
        # may increase at first
        region_percent = 2
    vel = np.sum(np.abs(track.velocity))
    if entering or exiting:
        region_percent = 2
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

    # score thumbs based on not being false positive having priority
    # then if sure of the prediction (above 80%) choose the most confidence
    # and then choose the one with more points
    def score(self):
        confidence_threshold = 80

        score = self.points
        score_offset = 100000

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
        # logging.info("%s score for %s %s %s is %s",self.track_id,self.points,self.predicted_tag,self.predicted_confidence, score)
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

    # keeps track of which id number we are up to.
    _track_id = 1

    # Percentage increase that is considered jitter, e.g. if a region gets
    # 30% bigger or smaller
    JITTER_THRESHOLD = 0.3
    # must change atleast 5 pixels to be considered for jitter
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
        """
        Creates a new Track.
        :param id: id number for track, if not specified is provided by an auto-incrementer
        """
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
        # our bounds over time
        self.bounds_history = []
        # number frames since we lost target.

        self.vel_x = []
        self.vel_y = []
        # the tag for this track
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
        # self.tracker = RegionTracker(
        #     self.get_id(), tracking_config, self.crop_rectangle
        #
        self.thumb_info = None
        self.score = None

    @property
    def id(self):
        return self._id

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

    def get_id(self):
        return self._id

    #
    def add_region(self, region):
        if self.prev_frame_num and region.frame_number:
            frame_diff = region.frame_number - self.prev_frame_num - 1
            for _ in range(frame_diff):
                self.add_blank_frame()
        self.tracker.add_region(region)
        self.bounds_history.append(region)
        # self.end_frame = region.frame_number
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

    def average_area(self):
        """Average mass of last 5 frames that weren't blank"""
        avg_area = 0
        count = 0
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
        avg_mass = 0
        count = 0
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
        """Maintains same bounds as previously, does not reset framce_since_target_seen counter"""
        if self.tracker:
            region = self.tracker.add_blank_frame()
        self.bounds_history.append(region)
        self.prev_frame_num = region.frame_number
        self.update_velocity()

    def set_end_s(self, fps):
        if len(self) == 0:
            self.end_s = self.start_s
            return
        self.end_s = (self.end_frame + 1) / fps

    def predicted_velocity(self):
        return self.tracker.predicted_velocity()

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
