from abc import ABC, abstractmethod
import logging
import numpy as np


class SlidingWindow:
    def __init__(self, shape, dtype):
        self.frames = [None] * shape
        # else:
        # self.frames = np.empty(shape, dtype)
        self.last_index = None
        self.size = len(self.frames)
        self.oldest_index = None
        self.non_ffc_index = None
        self.ffc = False

    def update_current_frame(self, frame, ffc=False):
        if self.last_index is None:
            self.oldest_index = 0
            self.last_index = 0
            if not ffc:
                self.non_ffc_index = self.oldest_index
        if not ffc and self.ffc:
            self.non_ffc_index = self.last_index

        self.frames[self.last_index] = frame
        self.ffc = ffc

    @property
    def current(self):
        if self.last_index is not None:
            return self.frames[self.last_index]
        return None

    def get_frames(self):
        if self.last_index is None:
            return []
        frames = []
        cur = self.oldest_index
        end_index = (self.last_index + 1) % self.size
        while len(frames) == 0 or cur != end_index:
            frames.append(self.frames[cur])
            cur = (cur + 1) % self.size
        return frames

    def get(self, i):
        i = i % self.size
        return self.frames[i]

    @property
    def oldest_nonffc(self):
        if self.non_ffc_index is not None:
            return self.frames[self.non_ffc_index]
        return None

    @property
    def oldest(self):
        if self.oldest_index is not None:
            return self.frames[self.oldest_index]
        return None

    def add(self, frame, ffc=False):
        if self.last_index is None:
            self.oldest_index = 0
            self.frames[0] = frame
            self.last_index = 0
            if not ffc:
                self.non_ffc_index = self.oldest_index
        else:
            new_index = (self.last_index + 1) % self.size
            if new_index == self.oldest_index:
                if self.oldest_index == self.non_ffc_index and not ffc:
                    self.non_ffc_index = (self.oldest_index + 1) % self.size
                self.oldest_index = (self.oldest_index + 1) % self.size
            self.frames[new_index] = frame
            self.last_index = new_index
        if not ffc and self.ffc:
            self.non_ffc_index = self.last_index
        self.ffc = ffc

    def reset(self):
        self.last_index = None
        self.oldest_index = None


class MotionDetector(ABC):
    def __init__(self, thermal_config, headers):
        self.movement_detected = False
        if thermal_config is not None:
            self.use_low_power_mode = thermal_config.recorder.use_low_power_mode
            self.rec_window = thermal_config.recorder.rec_window
            self.location_config = thermal_config.location
            self.use_sunrise = self.rec_window.use_sunrise_sunset()
        else:
            self.use_low_power_mode = False
            self.rec_window = None
            self.location_config = None
            self.use_sunrise = False
        self.num_frames = 0

        self.last_sunrise_check = None
        self.location = None
        self.sunrise = None
        self.sunset = None
        self.recording = False

        if self.use_sunrise:
            self.rec_window.set_location(
                *self.location_config.get_lat_long(use_default=True),
                self.location_config.altitude,
            )
        if self.rec_window:
            logging.info(
                "Recording window %s - %s ",
                self.rec_window.start.dt,
                self.rec_window.end.dt,
            )
        self.headers = headers

    @property
    def res_x(self):
        return self.headers.res_x

    @property
    def res_y(self):
        return self.headers.res_y

    @abstractmethod
    def process_frame(self, clipped_frame, received_at=None):
        """Tracker type IR or Thermal"""

    @abstractmethod
    def preview_frames(self):
        """Tracker type IR or Thermal"""

    @abstractmethod
    def get_recent_frame(self):
        """Tracker type IR or Thermal"""

    def can_record(self):
        return (
            self.rec_window.inside_window()
            if self.rec_window
            else True and not self.use_low_power_mode
        )

    @abstractmethod
    def disconnected(self):
        """Tracker type IR or Thermal"""

    @abstractmethod
    def calibrating(self):
        """Tracker type IR or Thermal"""

    @property
    @abstractmethod
    def background(self):
        """Tracker type IR or Thermal"""


class RunningMean:
    def __init__(self, data, window_size):
        self.running_mean = np.sum(data, axis=0, dtype=np.uint32)
        self.running_mean_frames = len(data)
        self.window_size = window_size

    def add(self, new_data, oldest_data):
        if self.running_mean_frames == self.window_size:
            self.running_mean -= oldest_data
            self.running_mean += new_data
        else:
            self.running_mean = self.running_mean + new_data
            self.running_mean_frames += 1

    def mean(self):
        return self.running_mean / self.running_mean_frames


class WeightedBackground:
    MEAN_FRAMES = 45

    def __init__(
        self, edge_pixels=1, res_x=160, res_y=120, weight_add=1, init_average=28000
    ):
        from rectangle import Rectangle

        self.crop_rectangle = Rectangle(
            edge_pixels, edge_pixels, res_x - 2 * edge_pixels, res_y - 2 * edge_pixels
        )
        self.thermal_window = SlidingWindow(45 + 1, "O")
        self.ffc_affected = False
        self.edge_pixels = edge_pixels
        self._background = None
        self.weight_add = weight_add
        self.background_weight = np.zeros(
            (res_y - edge_pixels * 2, res_x - edge_pixels * 2)
        )
        self.running_mean = None
        self.num_frames = 0
        # there is not much need to this as it gets updated after processing 1 frame
        # and can just calculate it from the background frame
        if init_average is not None:
            self.average = init_average

    def get_average(self):
        return self.average

    def process_frame(self, cptv_frame, ffc_affected):
        self.num_frames += 1
        prev_ffc = self.ffc_affected
        self.ffc_affected = ffc_affected
        self.thermal_window.add(cptv_frame, self.ffc_affected)
        oldest_thermal = self.thermal_window.oldest
        if oldest_thermal is not None:
            oldest_thermal = oldest_thermal.pix
        if self.running_mean is None:
            last_45 = self.thermal_window.get_frames()[: self.MEAN_FRAMES]
            last_45 = [f.pix for f in last_45]
            if len(last_45) > 0:
                self.running_mean = RunningMean(last_45, self.MEAN_FRAMES)

        else:
            self.running_mean.add(cptv_frame.pix, oldest_thermal)

        frame = np.int32(self.crop_rectangle.subimage(self.running_mean.mean()))
        if self._background is None:
            res_y, res_x = frame.shape
            self._background = np.empty(
                (res_y + self.edge_pixels * 2, res_x + self.edge_pixels * 2)
            )
            self._background[
                self.edge_pixels : res_y + self.edge_pixels,
                self.edge_pixels : res_x + self.edge_pixels,
            ] = frame
            self.average = np.average(frame)
            self.set_background_edges()
            return
        edgeless_back = self.crop_rectangle.subimage(self.background)
        new_background = np.where(
            edgeless_back < frame - self.background_weight,
            edgeless_back,
            frame,
        )
        # update weighting
        # weights could be adjusted to less while recording
        self.background_weight = np.where(
            edgeless_back < frame - self.background_weight,
            self.background_weight + self.weight_add,
            0,
        )
        back_changed = new_background != edgeless_back
        back_changed = np.any(back_changed == True)
        if back_changed:
            edgeless_back[:, :] = new_background
            old_temp = self.average
            self.average = int(round(np.average(edgeless_back)))
            if self.average != old_temp:
                logging.debug(
                    "MotionDetector temp threshold changed from {} to {} ".format(
                        old_temp,
                        self.average,
                    )
                )
            self.set_background_edges()

        if self.ffc_affected or prev_ffc:
            logging.debug("{} MotionDetector FFC".format(self.num_frames))
            self.movement_detected = False
            self.triggered = 0
            if prev_ffc:
                self.thermal_window.non_ffc_index = self.thermal_window.last_index

    def set_background_edges(self):
        for i in range(self.edge_pixels):
            self._background[i] = self._background[self.edge_pixels]
            self._background[-i - 1] = self._background[-self.edge_pixels - 1]
            self._background[:, i] = self._background[:, self.edge_pixels]
            self._background[:, -i - 1] = self._background[:, -1 - self.edge_pixels]

    @property
    def background(self):
        return self._background
