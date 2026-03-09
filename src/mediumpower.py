import socket

# from .headerinfo import HeaderInfo
import os
import time
import logging
import sys
import multiprocessing
import threading
from liteinterpreter import LiteInterpreter

# from config.thermalconfig import ThermalConfig
# from config.config import Config

fmt = "%(asctime)s %(process)d %(thread)s:%(levelname)7s %(message)s"

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
)
SOCKET_NAME = "/var/run/lepton-frames"
start = time.time()


DEBUG = False


def parse_cptv(cptv_file, frame_queue):
    from cptv_rs_python_bindings import CptvReader
    from cptv import Frame

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

    test = False
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
    global start
    while True:
        logging.info("waiting for a connection %s", time.time() - start)
        try:
            connection, client_address = sock.accept()
            connected = True
            logging.info("connection from %s", client_address)
            # log_event("camera-connected", {"type": "thermal"})
            medium_power(connection, frame_queue, processor)
        except KeyboardInterrupt:
            logging.info("\nCtrl+C pressed. Exiting gracefully.")
            break
        except Exception as ex:
            logging.error("Error with connection", exc_info=True)

        finally:
            # Clean up the connection
            try:
                connection.close()
            except:
                pass
        connected = False
        start = time.time()
    frame_queue.put(STOP_SIGNAL)
    processor.join()


def handle_headers(connection):
    headers = b""
    left_over = None
    while True:
        logging.info("Getting header info")
        data = connection.recv(4096)
        if not data:
            raise Exception("Disconnected from camera while getting headers")
        headers += data
        done = headers.find(b"\n\n")
        if done > -1:
            # logging.info("Headers %s done %s ", headers, done)
            # need the clear message
            left_over = headers[done + 2 :]
            headers = headers[:done]

            # ensure we handle the clear message
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


def medium_power(connection, frame_queue, processor):
    from cptv_rs_python_bindings import CptvStreamReader
    import zlib
    import numpy as np
    from cptv import Frame

    headers, extra_b = handle_headers(connection)
    stream_i = 0
    connection.settimeout(5)
    logging.info("Medium Power =======")
    while True:

        # wait for start message
        if extra_b is None:
            try:
                extra_b = connection.recv(headers.frame_size)
            except:
                logging.error("Couldnt get start", exc_info=True)
                extra_b = None
                time.sleep(1)
                continue
        start_index = extra_b.find(b"start\n\n")
        if start_index > -1:
            extra_b = extra_b[start_index + len("start\n\n") :]
        else:
            if len(extra_b) == 0:
                # disconnected
                return
            extra_b = None
            continue

        reader = CptvStreamReader()
        decompressor = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
        recording = False
        u8_data = None
        frame_i = 0
        read_header = False
        data = b""
        finished = False

        if DEBUG:
            logging.info(f"Writing raw bytes to /home/pi/streamed/raw{stream_i}.gz")
            f = open(f"/home/pi/streamed/raw{stream_i}.gz", "wb")
        byte_data = b""
        if extra_b is not None:
            data = extra_b
            if DEBUG:
                f.write(extra_b)
        stream_i += 1
        while not finished:
            try:
                byte_data = connection.recv(headers.frame_size)
                if len(byte_data) == 0:
                    # disconnected from socket
                    logging.info("Disconnected from socket")
                    if recording:
                        logging.error("Mid recording failed to receive more data")
                        frame_queue.put(CLEAR_SIGNAL)
                    return
            except:
                if recording:
                    logging.error("Mid recording failed to receive more data")
                    frame_queue.put(CLEAR_SIGNAL)
                    break
                time.sleep(1)
                continue

            if len(byte_data) > 0:
                clear_index = byte_data.find(b"clear")
                if clear_index > -1:
                    byte_data = byte_data[:clear_index]

                    logging.info("Received clear finished file")
                    finished = True
                    frame_queue.put(CLEAR_SIGNAL)
                    if DEBUG:
                        f.write(byte_data)
                        f.close()
                    # might have another start
                    extra_b = byte_data[clear_index + len("clear") :]
                else:
                    if DEBUG:
                        f.write(byte_data)
                    logging.debug(
                        "Adding new data %s to old data %s", len(byte_data), len(data)
                    )
                    data = data + byte_data

            if len(data) == 0:
                time.sleep(1)
                continue

            try:
                logging.debug("Decompressing %s", len(data))
                data, decompressed_chunk, read_header = decompress(
                    decompressor, data, read_header
                )
            except:
                # if this happens log it and then get the file from rp2040
                logging.error("Error decompressing ", exc_info=True)
                return
                # time.sleep(1)
                # continue

            if len(decompressed_chunk) == 0:
                continue
            recording = True
            if u8_data is None:
                u8_data = np.frombuffer(decompressed_chunk, dtype=np.uint8)
            else:
                # logging.info("Adding more u8 %s to existing %s", len(decompressed_chunk), len(u8_data))
                u8_data = np.concatenate(
                    (u8_data, np.frombuffer(decompressed_chunk, dtype=np.uint8)), axis=0
                )

            # logging.info("Loading frames wtih %s", len(u8_data))
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
                    logging.debug(
                        "Have %s bytes but need more to decompress a frame",
                        len(u8_data),
                    )
                    break
        logging.info(
            "Finished processing left over bytes are %s num frames %s",
            "None" if u8_data is None else len(u8_data),
            frame_i,
        )


import zlib
import io
import struct
import gzip


def decompress(decompressor, data, read_header=False):

    fp = io.BytesIO(data)
    if not read_header:
        result = gzip._read_gzip_header(fp)
        if result is None:
            raise Exception("No gzip header found")
            # logging.info("Couldn't read header")
            # return data, b"", read_header
        data = data[fp.tell() :]
        read_header = True
    try:
        decompressed = decompressor.decompress(data)
    except:
        logging.error("Error decompressing ", exc_info=True)
        return data, b"", read_header
    unused_data = decompressor.unused_data[8:].lstrip(b"\x00")

    # print("Tell is no0w ", fp.tell()," Unused data is " , len(decompressor.unused_data), " decompressed is ",len(decompressed))
    if not decompressor.eof or len(decompressor.unused_data) < 8:
        print("Reach eof")
        # 1/0
        return unused_data, decompressed, read_header
        raise EOFError(
            "Compressed file ended before the end-of-stream " "marker was reached"
        )
    crc, length = struct.unpack("<II", decompressor.unused_data[:8])

    if crc != zlib.crc32(decompressed):
        # not check this proparly so will always error
        logging.error("CRC error")
        return unused_data, decompressed, read_header

        raise Exception("CRC check failed")
    if length != (len(decompressed) & 0xFFFFFFFF):
        raise Exception("Incorrect length of data produced")
    return unused_data, decompressed, read_header


"""
HeaderInfo describes a thermal cameras specs.
When a thermal camera first connects to the socket it will send some header
information describing it's specs e.g. Resolution, Frame rate
"""

import yaml
import attr


@attr.s
class HeaderInfo:
    X_RESOLUTION = "ResX"
    Y_RESOLUTION = "ResY"
    FPS = "FPS"
    MODEL = "Model"
    BRAND = "Brand"
    PIXEL_BITS = "PixelBits"
    FRAME_SIZE = "FrameSize"
    SERIAL = "CameraSerial"
    FIRMWARE = "Firmware"
    medium_power = attr.ib(default=False)
    res_x = attr.ib(default=160)
    res_y = attr.ib(default=120)
    fps = attr.ib(default=9)
    brand = attr.ib(default="lepton")
    model = attr.ib(default="lepton3.5")
    frame_size = attr.ib(default=39040)
    pixel_bits = attr.ib(default=16)
    serial = attr.ib(default="12")
    firmware = attr.ib(default="12")

    @classmethod
    def parse_header(cls, raw_string):
        if raw_string == "medium":
            return cls(medium_power=True)
        raw = yaml.safe_load(raw_string)

        headers = cls(
            res_x=raw.get(HeaderInfo.X_RESOLUTION),
            res_y=raw.get(HeaderInfo.Y_RESOLUTION),
            fps=raw.get(HeaderInfo.FPS),
            brand=raw.get(HeaderInfo.BRAND),
            model=raw.get(HeaderInfo.MODEL),
            serial=raw.get(HeaderInfo.SERIAL),
            frame_size=raw.get(HeaderInfo.FRAME_SIZE),
            pixel_bits=raw.get(HeaderInfo.PIXEL_BITS),
            firmware=raw.get(HeaderInfo.FIRMWARE),
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
                    HeaderInfo.X_RESOLUTION,
                    HeaderInfo.Y_RESOLUTION,
                    HeaderInfo.FPS,
                    HeaderInfo.PIXEL_BITS,
                )
            )
        return True

    def as_dict(self):
        return attr.asdict(self)


CLEAR_SIGNAL = "clear"

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


def identify_last_frame(monitored_tracks, clip, load_model_thread):
    import numpy as np
    from trackprediction import TrackPrediction

    global last_frame_predicted

    active_tracks = get_active_tracks(clip)
    if len(active_tracks) == 0:
        logging.info("No active tracks %s", len(clip.active_tracks))
        return
    if load_model_thread.is_alive():
        logging.info("Waiting for model thread to finish")
        load_model_thread.join()
    if classifier is None:
        logging.info("Not classifying as couldn't load model")
        return
    track = active_tracks[0]
    start = time.time()
    pred_result = classifier.predict_recent_frames(
        clip,
        track,
    )
    if pred_result is not None:
        track_pred = monitored_tracks.setdefault(track.id, TrackPrediction(track.id))
        prediction, frames, _ = pred_result
        track_pred.classified_frame(frames, prediction[0])

        # predicted_as= classifier.labels[track_pred.best_label_index]
        logging.info(
            "Track %s is predicted as %s conf %s took %s track frames %s",
            track,
            classifier.labels[track_pred.best_label_index],
            round(track_pred.normalized_best_score() * 100),
            time.time() - start,
            len(track),
        )

        # do this in main loop
        # if dbus_service is not None:
        #     dbus_service.tracking(
        #         clip.id,
        #         track.id,
        #         track_pred.normalized_score,
        #         track.bounds_history[-1],
        #         True,
        #         track_pred.last_frame_classified,
        #         predicted_as,
        #         classifier.id,
        #     )
    else:
        logging.error("Pred is none for %s", track)
    return monitored_tracks


classifier = None


def load_model():
    global classifier
    logging.info("Loading tflite model")

    from pathlib import Path

    try:
        classifier = LiteInterpreter(
            Path("/home/pi/tflite/converted_model.tflite"), 1, False, False
        )
        # this way metadata is loaded
        classifier.load_model()
        logging.info("Loaded tflite model")

    except:
        logging.error("Could not load model", exc_info=True)


def run_classifier(frame_queue):
    from dbusservice import DbusService

    load_model_thread = threading.Thread(target=load_model)
    load_model_thread.start()
    # dont think we need this
    headers = {}
    frame_i = 0
    predict_every = 20
    # this might need to be stored and queryable as not all services that want these may be running yet
    dbus_service = None
    try:
        while True:
            logging.info("Making a new clip")
            monitored_tracks = {}

            track_extractor, clip = new_clip()
            logging.info("Waiting for frames")
            while True:
                frame = frame_queue.get()
                if isinstance(frame, str):
                    if frame == CLEAR_SIGNAL:
                        logging.info(
                            "PiClassifier received clear signal will start a new clip"
                        )
                        if dbus_service:

                            for track_id, track_pred in monitored_tracks.items():
                                predicted_as = classifier.labels[
                                    track_pred.best_label_index
                                ]
                                track = [
                                    track
                                    for track in clip.tracks
                                    if track.id == track_id
                                ][0]
                                dbus_service.tracking(
                                    clip.id,
                                    track.id,
                                    track_pred.normalized_score(),
                                    track.bounds_history[-1],
                                    False,
                                    track_pred.last_frame_classified,
                                    predicted_as,
                                    classifier.id,
                                )
                            dbus_service.recording(False)
                        else:
                            logging.error(
                                "Dbus service never got started and recording is now finished"
                            )

                    elif frame == STOP_SIGNAL:
                        logging.info("PiClassifier received stop signal")
                        return
                else:
                    frame, time_sent = frame
                    stale_tracks = track_extractor.process_frame(clip, frame)
                    if dbus_service and classifier is not None:
                        try:
                            dbus_service = DbusService(headers, classifier.labels)
                            dbus_service.recording(True)
                        except:
                            logging.error(
                                "Couldnt load dbus will try again ", exc_info=True
                            )
                    frame_i += 1

                    # remove stale tracks
                    if len(monitored_tracks) > 0 and dbus_service:
                        for track in stale_tracks:
                            if track.id in monitored_tracks:
                                track_pred = monitored_tracks[track.id]
                                predicted_as = classifier.labels[
                                    track_pred.best_label_index
                                ]

                                dbus_service.tracking(
                                    clip.id,
                                    track.id,
                                    track_pred.normalized_score(),
                                    track.bounds_history[-1],
                                    False,
                                    track_pred.last_frame_classified,
                                    predicted_as,
                                    classifier.id,
                                )
                                del monitored_tracks[track.id]

                    if frame_i % predict_every == 0:
                        logging.info(
                            "%s Predicting behind by %s ",
                            frame_i,
                            time.time() - time_sent,
                        )
                        identify_last_frame(monitored_tracks, clip, load_model_thread)

                if dbus_service is not None:
                    for track_id, track_pred in monitored_tracks.items():
                        predicted_as = classifier.labels[track_pred.best_label_index]
                        track = [
                            track
                            for track in clip.active_tracks
                            if track.id == track_id
                        ][0]
                        dbus_service.tracking(
                            clip.id,
                            track.id,
                            track_pred.normalized_score(),
                            track.bounds_history[-1],
                            True,
                            track_pred.last_frame_classified,
                            predicted_as,
                            classifier.id,
                        )

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
    ID = 1

    def __init__(self):
        # do something
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
        self.id = BasicClip.ID
        BasicClip.ID += 1

    def get_id(self):
        return self.id

    def add_frame(self, thermal, filtered, mask=None, ffc_affected=False):
        self.current_frame += 1
        if ffc_affected:
            self.ffc_frames.append(self.current_frame)

        f = self.frame_buffer.add_frame(
            thermal, filtered, mask, self.current_frame, ffc_affected
        )

        return f

    def get_frame(self, frame_number):
        return self.frame_buffer.get_frame(frame_number)

    def _add_active_track(self, track):
        self.active_tracks.add(track)
        self.tracks.append(track)


class FrameBuffer:
    """Stores entire clip in memory, required for some operations such as track exporting."""

    def __init__(
        self,
        keep_frames=True,
        max_frames=50,
    ):

        self.frames = []
        self.frames_by_frame_number = {}
        self.prev_frame = None

        self.max_frames = max_frames
        self.keep_frames = True if max_frames and max_frames > 0 else keep_frames
        self.current_frame_i = 0
        self.current_frame = None

    def add_frame(self, thermal, filtered, mask, frame_number, ffc_affected=False):
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
        """
        Empties buffer
        """
        self.frames = []
        self.frames_by_frame_number = {}

    def get_frame(self, frame_number):
        frame = None
        if frame_number in self.frames_by_frame_number:
            frame = self.frames_by_frame_number[frame_number]
        elif self.prev_frame and self.prev_frame.frame_number == frame_number:
            return self.prev_frame
        elif self.current_frame and self.current_frame.frame_number == frame_number:
            return self.current_frame
        assert (
            frame == None or frame.frame_number == frame_number
        ), f"{frame.frame_number} is not the same as requested {frame_number}"
        return frame


@attr.s(slots=True, eq=False)
class Frame:
    thermal = attr.ib()
    filtered = attr.ib()
    frame_number = attr.ib()
    ffc_affected = attr.ib(default=False)
    region = attr.ib(default=None)

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

    def get_channel(self, channel):
        if channel == 0:
            return self.thermal
        return self.filtered


if __name__ == "__main__":
    main()
