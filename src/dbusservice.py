import threading
import logging
import json
import numpy as np
import time
import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib

from dbus.mainloop.glib import DBusGMainLoop

DBUS_NAME = "org.cacophony.thermalrecorder"
DBUS_PATH = "/org/cacophony/thermalrecorder"


class Service(dbus.service.Object):
    def __init__(
        self,
        dbus,
        headers,
        labels,
    ):
        super().__init__(dbus, DBUS_PATH)
        self.headers = headers
        self.labels = labels

    @dbus.service.method(
        DBUS_NAME,
        in_signature="",
        out_signature="a{si}",
    )
    def CameraInfo(self):
        logging.debug("Serving headers %s", self.headers)
        headers = self.headers.as_dict()
        ir = headers.get("model") == "IR"
        for k, v in headers.items():
            try:
                headers[k] = int(v)
            except:
                headers[k] = 0
                pass
        headers["FPS"] = headers.get("fps", 9)
        headers["ResX"] = headers.get("res_x", 160)
        headers["ResY"] = headers.get("res_y", 120)
        if ir:
            headers["Model"] = 2
        else:
            headers["Model"] = 1
        logging.debug("Sending headers %s", headers)
        return headers

    @dbus.service.method(DBUS_NAME, signature="a{ias}")
    def ClassificationLabels(self):
        logging.info("Getting labels %s", self.labels)
        return self.labels

    @dbus.service.signal(DBUS_NAME, signature="iiaisiaiiibbis")
    def Tracking(
        self,
        clip_id,
        track_id,
        prediction,
        what,
        confidence,
        region,
        frame,
        mass,
        blank,
        tracking,
        last_prediction_frame,
        model_id,
    ):
        pass

    @dbus.service.signal(DBUS_NAME, signature="ii")
    def TrackFiltered(self, clip_id, track_id):
        pass

    @dbus.service.signal(DBUS_NAME, signature="xb")
    def Recording(self, timestamp, is_recording):
        pass

    @dbus.service.signal(DBUS_NAME)
    def ServiceStarted(self):
        pass


class DbusService:
    def __init__(self, headers, labels):
        DBusGMainLoop(set_as_default=True)
        dbus.mainloop.glib.threads_init()
        self.loop = GLib.MainLoop()
        self.t = threading.Thread(
            target=self.run_server,
            args=(
                headers,
                labels,
            ),
        )
        self.t.start()
        self.service = None

    def quit(self):
        self.loop.quit()

    def run_server(self, headers, labels):
        session_bus = dbus.SystemBus(mainloop=DBusGMainLoop())
        name = dbus.service.BusName(DBUS_NAME, session_bus)
        self.service = Service(session_bus, headers, labels)
        self.service.ServiceStarted()
        self.loop.run()

    def tracking(
        self,
        clip_id,
        track_id,
        prediction,
        region,
        tracking,
        last_prediction_frame,
        pred_label,
        model_id,
    ):
        logging.debug(
            "Tracking?  %s region %s prediction %s track %s",
            tracking,
            region,
            prediction,
            track_id,
        )
        if self.service is None:
            return
        if prediction is not None:
            predictions = prediction.copy()
            predictions = np.uint8(np.round(predictions * 100))
            best = np.argmax(predictions)
            self.service.Tracking(
                clip_id,
                track_id,
                predictions,
                pred_label,
                predictions[best],
                region.to_ltrb(),
                region.frame_number,
                region.mass,
                region.blank,
                tracking,
                last_prediction_frame,
                str(model_id),
            )
        else:
            self.service.Tracking(
                clip_id,
                track_id,
                [],
                "",
                0,
                region.to_ltrb(),
                region.frame_number,
                region.mass,
                region.blank,
                tracking,
                last_prediction_frame,
                "0",
            )

    def track_filtered(self, clip_id, track_id):
        if self.service is None:
            return
        self.service.TrackFiltered(clip_id, track_id)

    def recording(self, is_recording):
        if self.service is None:
            return
        self.service.Recording(np.int64(time.time()), is_recording)
