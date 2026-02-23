# cython: language_level=3
import socket
import os
import time
import logging
import sys
import multiprocessing
import threading

fmt = "%(asctime)s %(process)d %(thread)s:%(levelname)7s %(message)s"

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
)
SOCKET_NAME = "/var/run/lepton-frames"
start = time.time()


def parse_cptv(cptv_file, frame_queue):
    import cv2
    from cptv_rs_python_bindings import CptvReader
    from cptv import Frame
    import numpy as np

    reader = CptvReader(cptv_file)
    while True:
        frame = reader.next_frame()
        if frame is None:
            break
        py_frame = Frame(
            frame.pix,
            frame.time_on,
            frame.last_ffc_time,
            frame.temp_c,
            frame.last_ffc_temp_c,
        )
        frame_queue.put((py_frame, time.time()))
        time.sleep(1 / 9)
    frame_queue.put(STOP_SIGNAL)


def main():
    global connected

    thermal_config = None
    config = None
    frame_queue = multiprocessing.Queue()

    processor = get_processor(frame_queue)
    processor.start()

    test = True
    if test:
        print("Parsing test.cptv")
        parse_cptv("test.cptv", frame_queue)
        processor.join()
        return

    logging.info("Making sock")
    try:
        os.unlink(SOCKET_NAME)
    except OSError:
        if os.path.exists(SOCKET_NAME):
            raise
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    sock.bind(SOCKET_NAME)
    sock.settimeout(3 * 60)  # 3 minutes
    sock.listen(1)
    while True:
        logging.info("waiting for a connection %s", time.time() - start)
        try:
            connection, client_address = sock.accept()
            connected = True
            logging.info("connection from %s", client_address)
            handle_connection(
                connection, config, thermal_config, None, None, frame_queue, processor
            )
        except socket.timeout:
            print("TIMEOUT")
            return

        except Exception as ex:
            print("Error with connection", ex)
        finally:
            try:
                connection.close()
            except:
                pass
        connected = False


def handle_headers(connection):
    headers = b""
    left_over = None
    while True:
        logging.info("Getting header info")
        data = connection.recv(4096)
        logging.info("Received %s", len(data))
        if not data:
            raise Exception("Disconnected from camera while getting headers")
        headers += data
        done = headers.find(b"\n\n")
        if done > -1:
            left_over = headers[done + 2:]
            headers = headers[:done]

            if len(left_over) < 5:
                left_over += connection.recv(5 - len(left_over))

            if left_over[:5] == b"clear":
                left_over = left_over[5:]
            break
    header_s = headers.decode()
    logging.info("header is %s ", header_s)
    return HeaderInfo.parse_header(header_s), left_over


def get_processor(process_queue):
    p_processor = multiprocessing.Process(
        target=run_classifier,
        args=(process_queue,),
    )
    return p_processor


def medium_power(
    thermal_config, config, connection, headers, extra_b, frame_queue, processor
):
    from cptv_rs_python_bindings import CptvStreamReader
    import zlib
    import numpy as np

    logging.info(
        "Got header running medium power extra size %s: %s", len(extra_b), extra_b[:50]
    )

    reader = CptvStreamReader()
    decompressor = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
    u8_data = None
    cdef int frame_i = 0
    cdef bint read_header = False
    connection.settimeout(5)
    data = b""
    cdef bint finished = False
    logging.info(
        "Headers frame size is %s extra b size is %s", headers.frame_size, len(extra_b)
    )
    from cptv import Frame

    while not finished:
        byte_data = b""
        try:
            if extra_b is not None:
                byte_data = extra_b + connection.recv(headers.frame_size - len(extra_b))
                extra_b = None
            else:
                byte_data = connection.recv(headers.frame_size)
        except TimeoutError as e:
            logging.info("TImed out")
            time.sleep(1)
        except:
            logging.error("No data resetting data", exc_info=True)
            byte_data = b""
            data = b""
            time.sleep(1)
            continue

        if len(byte_data) > 0:
            clear_index = byte_data.find(b"clear")
            if clear_index > -1:
                byte_data = byte_data[:clear_index]

                logging.info("Received clear finished file")
                finished = True
                frame_queue.put(STOP_SIGNAL)
            else:
                data = data + byte_data

        try:
            data, decompressed_chunk, read_header = decompress(
                decompressor, data, read_header
            )
        except:
            logging.error("Error decompressing ", exc_info=True)
            time.sleep(1)
            if len(data) > 40000:
                logging.info("Failed")
            continue

        if len(decompressed_chunk) == 0:
            continue

        if u8_data is None:
            u8_data = np.frombuffer(decompressed_chunk, dtype=np.uint8)
        else:
            u8_data = np.concatenate(
                (u8_data, np.frombuffer(decompressed_chunk, dtype=np.uint8)), axis=0
            )

        while True:
            result = reader.next_frame_from_data(u8_data)
            if result is not None:
                frame, used = result
                u8_data = u8_data[used:]
                py_frame = Frame(
                    frame.pix,
                    frame.time_on,
                    frame.last_ffc_time,
                    frame.temp_c,
                    frame.last_ffc_temp_c,
                )
                frame_queue.put((py_frame, time.time()))

                frame_i += 1
            else:
                if len(u8_data) > 40000:
                    logging.info("Exiting have error")
                break
    processor.join()


def handle_connection(
    connection,
    config,
    thermal_config,
    process_queue,
    track_extractor,
    frame_queue,
    processor,
):
    headers, extra_b = handle_headers(connection)
    logging.info("Got headers %s", headers)

    return medium_power(
        thermal_config, config, connection, headers, extra_b, frame_queue, processor
    )


import zlib
import io
import struct
import gzip


def decompress(decompressor, data, bint read_header=False):
    cdef bint _read_header = read_header

    fp = io.BytesIO(data)
    if not _read_header:
        result = gzip._read_gzip_header(fp)
        if result is None:
            logging.info("Couldn't read header")
            return data, b"", _read_header
        data = data[fp.tell():]
        _read_header = True
        logging.info("Read header")
    try:
        decompressed = decompressor.decompress(data)
    except:
        logging.error("Error decompressing ", exc_info=True)
        return data, b"", _read_header
    unused_data = decompressor.unused_data[8:].lstrip(b"\x00")

    if not decompressor.eof or len(decompressor.unused_data) < 8:
        print("Reach eof")
        return unused_data, decompressed, _read_header
        raise EOFError(
            "Compressed file ended before the end-of-stream marker was reached"
        )
    crc, length = struct.unpack("<II", decompressor.unused_data[:8])

    if crc != zlib.crc32(decompressed):
        logging.error("CRC error")
        return unused_data, decompressed, _read_header

        raise Exception("CRC check failed")
    if length != (len(decompressed) & 0xFFFFFFFF):
        raise Exception("Incorrect length of data produced")
    return unused_data, decompressed, _read_header


"""
HeaderInfo describes a thermal cameras specs.
"""

import yaml

# Field name constants (used by parse_header)
_HDR_RES_X     = "ResX"
_HDR_RES_Y     = "ResY"
_HDR_FPS       = "FPS"
_HDR_MODEL     = "Model"
_HDR_BRAND     = "Brand"
_HDR_PIX_BITS  = "PixelBits"
_HDR_FRAME_SZ  = "FrameSize"
_HDR_SERIAL    = "CameraSerial"
_HDR_FIRMWARE  = "Firmware"


cdef class HeaderInfo:
    cdef public bint medium_power
    cdef public int res_x, res_y, fps, frame_size, pixel_bits
    cdef public object brand, model, serial, firmware

    # Class-level field-name constants kept for backward compatibility
    X_RESOLUTION = _HDR_RES_X
    Y_RESOLUTION = _HDR_RES_Y
    FPS          = _HDR_FPS
    MODEL        = _HDR_MODEL
    BRAND        = _HDR_BRAND
    PIXEL_BITS   = _HDR_PIX_BITS
    FRAME_SIZE   = _HDR_FRAME_SZ
    SERIAL       = _HDR_SERIAL
    FIRMWARE     = _HDR_FIRMWARE

    def __init__(self, medium_power=False, res_x=160, res_y=120, fps=9,
                 brand="lepton", model="lepton3.5", frame_size=39040,
                 pixel_bits=16, serial="12", firmware="12"):
        self.medium_power = medium_power
        self.res_x        = res_x if res_x is not None else 0
        self.res_y        = res_y if res_y is not None else 0
        self.fps          = fps if fps is not None else 0
        self.frame_size   = frame_size if frame_size is not None else 0
        self.pixel_bits   = pixel_bits if pixel_bits is not None else 0
        self.brand        = brand
        self.model        = model
        self.serial       = serial
        self.firmware     = firmware

    def __reduce__(self):
        return (
            self.__class__,
            (self.medium_power, self.res_x, self.res_y, self.fps,
             self.brand, self.model, self.frame_size, self.pixel_bits,
             self.serial, self.firmware),
        )

    @classmethod
    def parse_header(cls, raw_string):
        if raw_string == "medium":
            return cls(medium_power=True)
        raw = yaml.safe_load(raw_string)

        headers = cls(
            res_x=raw.get(_HDR_RES_X),
            res_y=raw.get(_HDR_RES_Y),
            fps=raw.get(_HDR_FPS),
            brand=raw.get(_HDR_BRAND),
            model=raw.get(_HDR_MODEL),
            serial=raw.get(_HDR_SERIAL),
            frame_size=raw.get(_HDR_FRAME_SZ),
            pixel_bits=raw.get(_HDR_PIX_BITS),
            firmware=raw.get(_HDR_FIRMWARE),
        )
        headers.firmware = str(headers.firmware)
        if headers.res_x and headers.res_y:
            if not headers.pixel_bits and headers.frame_size:
                headers.pixel_bits = int(
                    8 * headers.frame_size / (headers.res_x * headers.res_y)
                )
            elif not headers.frame_size and headers.pixel_bits:
                headers.frame_size = int(
                    headers.res_x * headers.res_y * headers.pixel_bits / 8
                )
        headers.validate()
        return headers

    def validate(self):
        if not (self.res_x and self.res_y and self.fps and self.pixel_bits):
            raise ValueError(
                "header info is missing a required field ({}, {}, {} and/or {})".format(
                    _HDR_RES_X, _HDR_RES_Y, _HDR_FPS, _HDR_PIX_BITS,
                )
            )
        return True

    def as_dict(self):
        return {
            "medium_power": self.medium_power,
            "res_x":        self.res_x,
            "res_y":        self.res_y,
            "fps":          self.fps,
            "brand":        self.brand,
            "model":        self.model,
            "frame_size":   self.frame_size,
            "pixel_bits":   self.pixel_bits,
            "serial":       self.serial,
            "firmware":     self.firmware,
        }


STOP_SIGNAL = "stop"
SKIP_SIGNAL = "skip"


def get_active_tracks(clip):
    """
    Gets current clips active_tracks and returns the top NUM_CONCURRENT_TRACKS order by priority
    """
    active_tracks = clip.active_tracks
    active_tracks = [track for track in active_tracks if len(track) >= 8]
    return active_tracks


last_frame_predicted = None


def identify_last_frame(clip, load_model_thread):
    import numpy as np

    global last_frame_predicted

    active_tracks = get_active_tracks(clip)
    if len(active_tracks) == 0:
        logging.info("No active tracks %s", len(clip.active_tracks))
        return
    if load_model_thread.is_alive():
        logging.info("Waiting for model thread to finish")
        load_model_thread.join()
    track = active_tracks[0]
    print("Track is ", track)
    start = time.time()
    pred_result = classifier.predict_recent_frames(
        clip,
        track,
    )
    cdef int max_i;
    if pred_result is not None:
        prediction, frames, mass = pred_result
        last_frame_predicted = frames[-1]
        prediction = prediction[0]
        max_i = np.argmax(prediction)
        logging.info(
            "Track %s is predicted as %s conf %s took %s track frames %s",
            track,
            classifier.labels[max_i],
            round(prediction[max_i] * 100),
            time.time() - start,
            len(track),
        )
    else:
        logging.error("Pred is none for %s", track)


classifier = None


def load_model():
    global classifier
    logging.info("Loading tflite model")

    from pathlib import Path

    classifier = LiteInterpreter(Path("./tflite/converted_model.tflite"), False, True)
    logging.info("Loaded tflite model")


def run_classifier(frame_queue):
    load_model_thread = threading.Thread(target=load_model)
    load_model_thread.start()

    cdef int frame_i = 0
    cdef int predict_every = 20
    try:
        logging.info("Loading clip")

        track_extractor, clip = new_clip()
        logging.info("Waiting for frames")
        while True:
            frame = frame_queue.get()
            if isinstance(frame, str):
                if frame == STOP_SIGNAL:
                    logging.info("PiClassifier received stop signal")
            else:
                frame, time_sent = frame
                track_extractor.process_frame(clip, frame)
                frame_i += 1
                if frame_i % predict_every == 0:
                    logging.info(
                        "%s Predicting behind by %s ", frame_i, time.time() - time_sent
                    )
                    identify_last_frame(clip, load_model_thread)
    except:
        logging.error("Error running classifier restarting ..", exc_info=True)

        return


class DefaultTracking:
    pass


def init_trackers():
    from cliptrackextractor import ClipTrackExtractor

    default_tracking = DefaultTracking()
    default_tracking.edge_pixels = 1
    default_tracking.frame_padding = 4
    default_tracking.min_dimension = 0
    default_tracking.track_smoothing = False
    default_tracking.denoise = False
    default_tracking.high_quality_optical_flow = False
    default_tracking.max_tracks = None
    default_tracking.filters = {
        "track_overlap_ratio": 0.5,
        "min_duration_secs": 0,
        "track_min_offset": 4.0,
        "track_min_mass": 2.0,
        "moving_vel_thresh": 4,
    }
    default_tracking.areas_of_interest = {
        "min_mass": 4.0,
        "pixel_variance": 2.0,
        "cropped_regions_strategy": "cautious",
    }

    default_tracking.aoi_min_mass = 4.0
    default_tracking.aoi_pixel_variance = 2.0
    default_tracking.cropped_regions_strategy = "cautious"
    default_tracking.track_min_offset = 4.0
    default_tracking.track_min_mass = 2.0
    default_tracking.track_overlap_ratio = 0.5
    default_tracking.min_duration_secs = 0
    default_tracking.min_tag_confidence = 0.8
    default_tracking.enable_track_output = True
    default_tracking.moving_vel_thresh = 4
    default_tracking.min_moving_frames = 2
    default_tracking.max_blank_percent = 30
    default_tracking.max_mass_std_percent = 0.55
    default_tracking.max_jitter = 20
    default_tracking.tracker = "RegionTracker"
    default_tracking.type = "thermal"
    default_tracking.params = {
        "base_distance_change": 450,
        "min_mass_change": 20,
        "restrict_mass_after": 1.5,
        "mass_change_percent": 0.55,
        "max_distance": 2000,
        "max_blanks": 18,
        "velocity_multiplier": 2,
        "base_velocity": 2,
    }
    default_tracking.filter_regions_pre_match = True
    default_tracking.min_hist_diff = None

    track_extractor = ClipTrackExtractor(default_tracking)
    return track_extractor


def new_clip():
    track_extractor = init_trackers()

    headers = HeaderInfo()
    clip = BasicClip()
    clip.crop_rectangle = track_extractor.background_alg.crop_rectangle

    return track_extractor, clip


class BasicClip:
    def __init__(self):
        self.res_x = 160
        self.res_y = 120
        self.active_tracks = set()
        self.tracks = []
        self.current_frame = -1
        self.ffc_affected = False
        self.ffc_frames = []
        self.frame_buffer = FrameBuffer()
        self.region_history = []
        self.crop_rectangle = None
        self.frames_per_second = 9

    def get_id(self):
        return 1

    def add_frame(self, thermal, filtered, mask=None, ffc_affected=False):
        self.current_frame += 1
        if ffc_affected:
            self.ffc_frames.append(self.current_frame)

        f = self.frame_buffer.add_frame(
            thermal, filtered, mask, self.current_frame, ffc_affected
        )

        return f

    def get_frame(self, int frame_number):
        return self.frame_buffer.get_frame(frame_number)

    def _add_active_track(self, track):
        self.active_tracks.add(track)
        self.tracks.append(track)


class FrameBuffer:
    """Stores entire clip in memory."""

    def __init__(
        self,
        keep_frames=True,
        int max_frames=50,
    ):
        self.frames = []
        self.frames_by_frame_number = {}
        self.prev_frame = None

        self.max_frames = max_frames
        self.keep_frames = True if max_frames and max_frames > 0 else keep_frames
        self.current_frame_i = 0
        self.current_frame = None

    def add_frame(self, thermal, filtered, mask, int frame_number, ffc_affected=False):
        self.prev_frame = self.current_frame
        frame = Frame(thermal, filtered, frame_number, ffc_affected=ffc_affected)
        self.current_frame = frame

        if self.max_frames and len(self.frames) == self.max_frames:
            del self.frames_by_frame_number[self.frames[0].frame_number]
            del self.frames[0]
        self.frames.append(frame)
        self.frames_by_frame_number[frame.frame_number] = frame
        return frame

    def reset(self):
        """Empties buffer"""
        self.frames = []
        self.frames_by_frame_number = {}

    def get_frame(self, int frame_number):
        frame = None
        if frame_number in self.frames_by_frame_number:
            frame = self.frames_by_frame_number[frame_number]
        elif self.prev_frame and self.prev_frame.frame_number == frame_number:
            return self.prev_frame
        elif self.current_frame and self.current_frame.frame_number == frame_number:
            return self.current_frame
        assert (
            frame is None or frame.frame_number == frame_number
        ), f"{frame.frame_number} is not the same as requested {frame_number}"
        return frame


cdef class Frame:
    cdef public object thermal
    cdef public object filtered
    cdef public int frame_number
    cdef public bint ffc_affected
    cdef public object region

    def __init__(self, thermal, filtered, frame_number, ffc_affected=False, region=None):
        self.thermal      = thermal
        self.filtered     = filtered
        self.frame_number = frame_number
        self.ffc_affected = ffc_affected
        self.region       = region

    def __reduce__(self):
        return (
            self.__class__,
            (self.thermal, self.filtered, self.frame_number,
             self.ffc_affected, self.region),
        )

    def crop_by_region(self, region):
        import numpy as np

        thermal = region.subimage(self.thermal)
        filtered = region.subimage(self.filtered)
        frame = Frame(
            np.float32(thermal),
            np.float32(filtered),
            self.frame_number,
            ffc_affected=self.ffc_affected,
            region=region,
        )
        return frame

    def resize_with_aspect(
        self,
        dim,
        crop_rectangle,
        keep_edge=True,
        edge_offset=(0, 0, 0, 0),
        original_region=None,
    ):
        from tools import resize_and_pad

        self.thermal = resize_and_pad(
            self.thermal,
            dim,
            self.region,
            crop_rectangle,
            keep_edge=keep_edge,
            edge_offset=edge_offset,
            original_region=original_region,
        )

        self.filtered = resize_and_pad(
            self.filtered,
            dim,
            self.region,
            crop_rectangle,
            keep_edge=keep_edge,
            pad=0,
            edge_offset=edge_offset,
            original_region=original_region,
        )

    def get_channel(self, int channel):
        if channel == 0:
            return self.thermal
        return self.filtered


class LiteInterpreter:
    TYPE = "TFLite"

    def __init__(self, model_file, run_over_network=False, load_model=True):
        self.model_file = model_file
        self.run_over_network = run_over_network
        self.load_json()
        if run_over_network or not load_model:
            return
        self.load_model()

    def load_json(self):
        """Loads model and parameters from file."""
        from pathlib import Path
        import json

        filename = Path(self.model_file)
        filename = filename.with_suffix(".json")

        logging.info("Loading metadata from %s", filename)
        metadata = json.load(open(filename, "r"))
        self.version = metadata.get("version", None)
        self.labels = metadata["labels"]
        self.mapped_labels = metadata.get("mapped_labels")

    def load_model(self):
        from ai_edge_litert.interpreter import Interpreter

        model_name = self.model_file.with_suffix(".tflite")
        self.interpreter = Interpreter(str(model_name))

        self.interpreter.allocate_tensors()

        self.output = self.interpreter.get_output_details()[0]
        self.input = self.interpreter.get_input_details()[0]

    def predict_recent_frames(self, clip, track):
        samples = frame_samples(clip, track)
        frames, preprocessed, mass = preprocess_segments(clip, track, samples)
        if preprocessed is None or len(preprocessed) == 0:
            return None

        import numpy as np
        preprocessed = np.expand_dims(preprocessed, axis=0)
        logging.info("Predicting on %s %s", frames, preprocessed.shape)

        prediction = self.predict(preprocessed)
        return prediction, frames, mass

    def predict(self, input_x):
        import numpy as np

        if self.run_over_network:
            return self.predict_over_network(np.float32(input_x))
        input_x = np.float32(input_x)
        preds = []
        for data in input_x:
            self.interpreter.set_tensor(self.input["index"], data[np.newaxis, :])
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output["index"])
            preds.append(pred[0])
        return preds

    def shape(self):
        return 1, self.input["shape"]


def frame_samples(clip, track, int num_frames=25):
    regions = track.bounds_history[-50:]

    cdef int start_index = 0
    for r in regions:
        if r.frame_number >= clip.current_frame - 50:
            break
        start_index += 1

    logging.info(
        "Getting frame samples current frame %s starting at %s frame# %s",
        clip.current_frame,
        start_index,
        regions[start_index].frame_number,
    )
    regions = regions[start_index:]
    regions = [
        region
        for region in regions
        if region.mass > 0
        and region.frame_number not in clip.ffc_frames
        and not region.blank
        and region.width > 0
        and region.height > 0
    ]

    import numpy as np

    rng = np.random.default_rng()
    samples = rng.choice(
        np.array(regions), size=min(len(regions), num_frames), replace=False
    )
    return samples


def preprocess_segments(clip, track, samples):
    import numpy as np
    from tools import preprocess_movement, normalize

    frame_temp_medians = {}
    cdef bint clip_thermals_at_zero = True
    filtered_norm_limits = get_limits(clip, track)
    frames = []
    cdef double mass = 0
    samples = sorted(samples, key=lambda sample: sample.frame_number)

    for region in samples:
        mass += region.mass

        frame_number = region.frame_number
        frame = clip.get_frame(frame_number)
        frame_temp_medians[frame_number] = np.median(frame.thermal)

        if clip_thermals_at_zero:
            sub_thermal = region.subimage(frame.thermal)
            sub_thermal = (
                np.float32(sub_thermal) - frame_temp_medians[region.frame_number]
            )
            if np.median(sub_thermal) <= 0:
                clip_thermals_at_zero = False
        cropped_frame = frame.crop_by_region(region)
        cropped_frame.thermal -= frame_temp_medians[frame_number]
        cropped_frame.resize_with_aspect((32, 32), clip.crop_rectangle)
        frames.append(cropped_frame)
    mass = mass / len(samples)
    preprocessed = []

    for frame in frames:
        frame.filtered, stats = normalize(
            frame.filtered,
            min=filtered_norm_limits[0],
            max=filtered_norm_limits[1],
            new_max=255,
        )
        frame.thermal, _ = normalize(frame.thermal, new_max=255)
        preprocessed.append(frame)
    preprocessed = preprocess_movement(preprocessed)

    if frames is None:
        logging.warn("No frames to predict on")
    preprocessed = np.array(preprocessed)
    return [region.frame_number for region in samples], preprocessed, mass


def get_limits(clip, track):
    import numpy as np

    min_diff = None
    cdef double max_diff = 0

    for region in reversed(track.bounds_history):
        if region.blank:
            continue
        if region.width == 0 or region.height == 0:
            logging.warn(
                "No width or height for frame %s region %s",
                region.frame_number,
                region,
            )
            continue
        f = clip.get_frame(region.frame_number)
        if region.blank or region.width <= 0 or region.height <= 0 or f is None:
            continue

        diff_frame = region.subimage(f.filtered)

        new_max = np.amax(diff_frame)
        new_min = np.amin(diff_frame)
        if min_diff is None or new_min < min_diff:
            min_diff = new_min
        if new_max > max_diff:
            max_diff = new_max

    filtered_norm_limits = (min_diff, max_diff)
    return filtered_norm_limits


if __name__ == "__main__":
    main()
