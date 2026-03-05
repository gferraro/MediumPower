import numpy as np


class TrackPrediction:
    """
    Class to hold the information about the predicted class of a track.
    Predictions are recorded for every frame, and methods provided for extracting the final predicted class of the
    track.
    """

    def __init__(self, track_id):
        self.track_id = track_id
        self.last_frame_classified = None
        self.num_frames_classified = 0
        self.class_best_score = None
        self.normalized = False

    def classified_frame(self, frame_numbers, predictions):
        self.last_frame_classified = frame_numbers[-1]
        self.num_frames_classified += 1

        if self.class_best_score is None:
            self.class_best_score = predictions
        else:
            self.class_best_score += predictions

    def normalized_best_score(self):
        # assume predictions in the range 0- 100
        return self.class_best_score[self.best_label_index] / self.num_frames_classified

    def normalized_score(self):
        # assume predictions in the range 0- 100
        return self.class_best_score / self.num_frames_classified

    def normalize_score(self):
        self.class_best_score = self.normalizedscore()
        self.normalized = True

    @property
    def best_label_index(self):
        if self.class_best_score is None:
            return None
        return np.argmax(self.class_best_score)
