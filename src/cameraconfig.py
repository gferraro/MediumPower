from pathlib import Path
import attr
import toml
import portalocker
import os

# from .timewindow import RelAbsTime, TimeWindow

CONFIG_FILENAME = "config.toml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc/cacophony")]


class LockSafeConfig:
    def __init__(self, filename):
        self.lock_file = filename + ".lock"
        self.filename = filename
        self.f = None
        self.lock = portalocker.Lock(
            self.lock_file, "r", flags=portalocker.LOCK_SH, timeout=1
        )
        if not os.path.exists(self.lock_file):
            f = open(self.lock_file, "w+")
            f.close()

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        self.lock.acquire()
        self.f = open(self.filename)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.f.close()
        finally:
            self.lock.release()


@attr.s
class CameraMotionConfig:
    temp_thresh = attr.ib()
    delta_thresh = attr.ib()
    count_thresh = attr.ib()
    frame_compare_gap = attr.ib()
    one_diff_only = attr.ib()
    trigger_frames = attr.ib()
    edge_pixels = attr.ib()
    warmer_only = attr.ib()
    dynamic_thresh = attr.ib()
    run_classifier = attr.ib(default=False)
    bluetooth_beacons = attr.ib(default=False)
    tracking_events = attr.ib(default=False)
    do_tracking = attr.ib(default=False)
    postprocess = attr.ib(default=False)
    postprocess_events = attr.ib(default=False)

    @classmethod
    def defaults_for(cls, model):
        if model == "lepton3.5":
            return cls(
                temp_thresh=28000,
                delta_thresh=150,
                count_thresh=3,
                frame_compare_gap=45,
                one_diff_only=True,
                trigger_frames=2,
                edge_pixels=1,
                warmer_only=True,
                dynamic_thresh=True,
                do_tracking=False,
            )
        else:
            return cls(
                temp_thresh=2750,
                delta_thresh=50,
                count_thresh=3,
                frame_compare_gap=45,
                one_diff_only=True,
                trigger_frames=2,
                edge_pixels=1,
                warmer_only=True,
                dynamic_thresh=True,
                do_tracking=False,
            )

    @classmethod
    def load(cls, motion, model=None):
        default = CameraMotionConfig.defaults_for(model)
        motion = cls(
            temp_thresh=motion.get("temp-thresh", default.temp_thresh),
            delta_thresh=motion.get("delta-thresh", default.delta_thresh),
            count_thresh=motion.get("count-thresh", default.count_thresh),
            frame_compare_gap=motion.get(
                "frame-compare-gap", default.frame_compare_gap
            ),
            one_diff_only=motion.get("use-one-diff-only", default.one_diff_only),
            trigger_frames=motion.get("trigger-frames", default.trigger_frames),
            edge_pixels=motion.get("edge-pixels", default.edge_pixels),
            warmer_only=motion.get("warmer-only", default.warmer_only),
            dynamic_thresh=motion.get("dynamic-thresh", default.dynamic_thresh),
            run_classifier=motion.get("run-classifier", default.run_classifier),
            bluetooth_beacons=motion.get(
                "bluetooth-beacons", default.bluetooth_beacons
            ),
            tracking_events=motion.get("tracking-events", default.tracking_events),
            do_tracking=motion.get("do-tracking", default.do_tracking),
            postprocess=motion.get("postprocess", default.postprocess),
            postprocess_events=motion.get(
                "postprocess-events", default.postprocess_events
            ),
        )
        return motion

    def as_dict(self):
        return attr.asdict(self)


@attr.s
class RecorderConfig:
    preview_secs = attr.ib()
    min_secs = attr.ib()
    max_secs = attr.ib()
    output_dir = attr.ib()
    disable_recordings = attr.ib()
    constant_recorder = attr.ib()
    use_low_power_mode = attr.ib()
    min_disk_space_mb = attr.ib()
    instant_classify = attr.ib()

    @classmethod
    def load(cls, recorder):
        return cls(
            constant_recorder=recorder.get("constant-recorder", False),
            disable_recordings=recorder.get("disable-recordings", False),
            min_secs=recorder.get("min-secs", 5),
            max_secs=recorder.get("max-secs", 600),
            preview_secs=recorder.get("preview-secs", 5),
            min_disk_space_mb=recorder.get("min-disk-space-mb", 200),
            output_dir=recorder.get("output-dir", "/var/spool/cptv"),
            use_low_power_mode=recorder.get("use-low-power-mode", False),
            instant_classify=recorder.get("instant-classify", False),
        )


@attr.s
class DeviceSetup:
    ir = attr.ib(default=False)
    trap_size = attr.ib(default=None)
    # S or L for small or large

    @classmethod
    def load(cls, device):
        size = device.get("trap-size", "L")
        if size is not None:
            size = size.upper()
        return cls(ir=device.get("ir", False), trap_size=size)


@attr.s
class DeviceConfig:
    device_id = attr.ib()
    name = attr.ib()

    @classmethod
    def load(cls, device):
        return cls(
            name=device.get("name"),
            device_id=device.get("id"),
        )


@attr.s
class ThermalConfig:
    motion = attr.ib()
    recorder = attr.ib()
    device = attr.ib()
    location = attr.ib()
    device_setup = attr.ib()
    config_file = attr.ib()

    @classmethod
    def load_from_file(cls, filename=None, model=None):
        if not filename:
            filename = ThermalConfig.find_config()
        with LockSafeConfig(filename) as stream:
            return cls.load_from_stream(filename, stream, model)

    @classmethod
    def load_from_stream(cls, filename, stream, model=None):
        raw = toml.load(stream)
        if raw is None:
            raw = {}

        location_config = LocationConfig.load(raw.get("location", {}))
        return cls(
            config_file=filename,
            motion=CameraMotionConfig.load(raw.get("thermal-motion", {}), model),
            recorder=RecorderConfig.load(raw.get("thermal-recorder", {})),
            device=DeviceConfig.load(raw.get("device", {})),
            device_setup=DeviceSetup.load(raw.get("device-setup", {})),
            location=location_config,
        )

    def validate(self):
        return True

    @staticmethod
    def find_config():
        for directory in CONFIG_DIRS:
            p = directory / CONFIG_FILENAME
            if p.is_file():
                return str(p)
        raise FileNotFoundError(
            "No configuration file found.  Looking for file named '{}' in dirs {}".format(
                CONFIG_FILENAME, CONFIG_DIRS
            )
        )


import attr


@attr.s
class LocationConfig:
    DEFAULT_LAT = -43.5321
    DEFAULT_LONG = 172.6362

    latitude = attr.ib()
    longitude = attr.ib()
    loc_timestamp = attr.ib()
    altitude = attr.ib()
    accuracy = attr.ib()

    @classmethod
    def load(cls, raw):
        return cls(
            latitude=raw.get("latitude", 0),
            longitude=raw.get("longitude", 0),
            loc_timestamp=raw.get("timestamp"),
            altitude=raw.get("altitude"),
            accuracy=raw.get("accuracy"),
        )

    def get_lat_long(self, use_default=False):
        lat = self.latitude
        lng = self.longitude
        if use_default and lat == 0:
            lat = LocationConfig.DEFAULT_LAT
        if use_default and lng == 0:
            lng = LocationConfig.DEFAULT_LONG
        return (lat, lng)
