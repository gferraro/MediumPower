import logging


class LiteInterpreter:
    TYPE = "TFLite"

    def __init__(self, model_file, id, run_over_network=False, load_model=True):
        self.model_file = model_file
        self.run_over_network = run_over_network
        self.load_json()
        self.id = id

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

        self.interpreter.allocate_tensors()  # Needed before execution!

        self.output = self.interpreter.get_output_details()[
            0
        ]  # Model has single output.

        self.input = self.interpreter.get_input_details()[0]  # Model has single input.
        # self.preprocess_fn = self.get_preprocess_fn()
        # inc3_preprocess

    # use when predicting as tracks are being tracked i.e not finished yet
    def predict_recent_frames(self, clip, track):
        samples = frame_samples(clip, track)
        frames, preprocessed, mass = preprocess_segments(clip, track, samples)
        if preprocessed is None or len(preprocessed) == 0:
            return None

        import numpy as np

        preprocessed = np.expand_dims(preprocessed, axis=0)
        logging.debug("Predicting on %s %s", frames, preprocessed.shape)

        prediction = self.predict(preprocessed)
        return prediction, frames, mass

    def predict(self, input_x):
        import numpy as np

        if self.run_over_network:
            return self.predict_over_network(np.float32(input_x))
        input_x = np.float32(input_x)
        preds = []
        # only works on input of 1
        for data in input_x:
            self.interpreter.set_tensor(self.input["index"], data[np.newaxis, :])
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output["index"])
            preds.append(pred[0])
        return preds

    def shape(self):
        return 1, self.input["shape"]


def frame_samples(clip, track, num_frames=25):
    regions = track.bounds_history[-50:]

    start_index = 0
    for r in regions:
        # only have 50 frames in memory
        if r.frame_number >= clip.current_frame - 50:
            break
        start_index += 1

    logging.debug(
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

    # Create a Generator instance (seed is optional for reproducibility)
    rng = np.random.default_rng()
    samples = rng.choice(
        np.array(regions), size=min(len(regions), num_frames), replace=False
    )
    return samples


def preprocess_segments(clip, track, samples):
    import numpy as np
    from tools import preprocess_movement, normalize

    frame_temp_medians = {}
    clip_thermals_at_zero = True
    filtered_norm_limits = get_limits(clip, track)
    frames = []
    mass = 0
    samples = sorted(samples, key=lambda sample: sample.frame_number)

    for region in samples:
        mass += region.mass

        frame_number = region.frame_number
        frame = clip.get_frame(frame_number)
        frame_temp_medians[frame_number] = np.median(frame.thermal)

        if clip_thermals_at_zero:
            # check that we have nice values other wise allow negatives when normalizing
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

    # import cv2
    # display = np.uint8(preprocessed[:,:,2])
    # cv2.imshow("f",display)
    # cv2.waitKey(0)
    if frames is None:
        logging.warn("No frames to predict on")
    preprocessed = np.array(preprocessed)
    return [region.frame_number for region in samples], preprocessed, mass


def get_limits(clip, track):
    min_diff = None
    max_diff = 0
    import numpy as np

    filtered_norm_limits = None
    for region in reversed(track.bounds_history):
        if region.blank:
            continue
        if region.width == 0 or region.height == 0:
            logging.warn(
                "No width or height for frame %s regoin %s",
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
