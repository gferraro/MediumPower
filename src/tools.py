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


def normalize(data, min=None, max=None, new_max=1):
    """
    Normalize an array so that the values range from 0 -> new_max
    Returns normalized array, stats tuple (Success, min used, max used)
    """
    import numpy as np

    if data.size == 0:
        return np.zeros((data.shape)), (False, None, None)
    if max is None:
        max = np.amax(data)
    if min is None:
        min = np.amin(data)
    if max == min:
        if max == 0:
            return np.zeros((data.shape)), (False, max, min)
        data = data / max
        return data, (True, max, min)

    data = new_max * (np.float32(data) - min) / (max - min)
    return data, (True, max, min)


def detect_objects(image, otsus=False, threshold=30, kernel=(15, 15)):
    import numpy as np
    import cv2

    image = np.uint8(image)
    image = cv2.GaussianBlur(image, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU
    _, image = cv2.threshold(image, threshold, 255, flags)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cv2.connectedComponentsWithStats(image)




def resize_cv(image, dim, interpolation, extra_h=0, extra_v=0):
    import cv2
    import numpy as np
    return cv2.resize(
        np.float32(image),
        dsize=(dim[0] + extra_h, dim[1] + extra_v),
        interpolation=interpolation,
    )


def resize_and_pad(
    frame,
    new_dim,
    region,
    crop_region,
    keep_edge=False,
    pad=None,
    interpolation=None,
    extra_h=0,
    extra_v=0,
    edge_offset=(0, 0, 0, 0),
    original_region=None,
):
    import cv2
    import numpy as np
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR

    scale_percent = (new_dim[:2] / np.array(frame.shape[:2])).min()
    width = round(frame.shape[1] * scale_percent)
    height = round(frame.shape[0] * scale_percent)
    width = max(width, 1)
    height = max(height, 1)

    width = min(width, new_dim[0])
    height = min(height, new_dim[1])

    if len(frame.shape) == 3:
        resize_dim = (width, height, frame.shape[2])
    else:
        resize_dim = (width, height)
    if pad is None:
        pad = np.min(frame)
    if original_region is None:
        original_region = region
    resized = np.full(new_dim, pad, dtype=frame.dtype)
    offset_x = 0
    offset_y = 0
    frame_resized = resize_cv(frame, resize_dim, interpolation=interpolation)
    frame_height, frame_width = frame_resized.shape[:2]
    offset_x = (new_dim[1] - frame_width) // 2
    offset_y = (new_dim[0] - frame_height) // 2
    if keep_edge and crop_region is not None:
        if original_region.left <= crop_region.left:
            offset_x = min(edge_offset[0], new_dim[1] - frame_width)
        elif original_region.right >= crop_region.right:
            offset_x = (new_dim[1] - edge_offset[2]) - frame_width
            offset_x = max(offset_x, 0)
        if original_region.top <= crop_region.top:
            offset_y = min(edge_offset[1], new_dim[0] - frame_height)

        elif original_region.bottom >= crop_region.bottom:
            offset_y = new_dim[0] - frame_height - edge_offset[3]
            offset_y = max(offset_y, 0)

    if len(resized.shape) == 3:
        resized[
            offset_y : offset_y + frame_height, offset_x : offset_x + frame_width, :
        ] = frame_resized
    else:
        resized[
            offset_y : offset_y + frame_height,
            offset_x : offset_x + frame_width,
        ] = frame_resized
    return resized


def eucl_distance_sq(first, second):
    first_sq = first[0] - second[0]
    first_sq = first_sq * first_sq
    second_sq = first[1] - second[1]
    second_sq = second_sq * second_sq

    return first_sq + second_sq


def preprocess_movement(
    preprocess_frames,
    frames_per_row=5,
    frame_size=32,
    channels=[0, 1, 1],
    preprocess_fn=None,
    sample=None,
):
    import numpy as np
    from tools import square_clip

    frame_types = {}
    data = []
    frame_samples = list(np.arange(len(preprocess_frames)))
    print("Samples are ",len(preprocess_frames))
    if len(preprocess_frames) < frames_per_row * 5:
        extra_samples = np.random.choice(
            frame_samples, frames_per_row * 5 - len(preprocess_frames)
        )
        frame_samples.extend(extra_samples)
        frame_samples.sort()
    for channel in channels:

        if channel in frame_types:
            data.append(frame_types[channel])
            continue
        channel_segment = [frame.get_channel(channel) for frame in preprocess_frames]
        channel_data, success = square_clip(
            channel_segment,
            frames_per_row,
            (frame_size, frame_size),
            frame_samples,
            normalize=False,
        )
        # already done normalization

        if not success:
            return None
        data.append(channel_data)
        frame_types[channel] = channel_data
    data = np.stack(data, axis=2)

    return np.float32(data)


def square_clip(data, frames_per_row, tile_dim, frame_samples, normalize=True):
    # lay each frame out side by side in rows
    import numpy as np

    new_frame = np.zeros((frames_per_row * tile_dim[0], frames_per_row * tile_dim[1]))
    i = 0
    success = False
    for x in range(frames_per_row):
        for y in range(frames_per_row):
            frame = data[frame_samples[i]]
            if normalize:
                frame, stats = normalize(frame, new_max=255)
                if not stats[0]:
                    continue
            success = True
            new_frame[
                x * tile_dim[0] : (x + 1) * tile_dim[0],
                y * tile_dim[1] : (y + 1) * tile_dim[1],
            ] = np.float32(frame)
            i += 1

    return new_frame, success
