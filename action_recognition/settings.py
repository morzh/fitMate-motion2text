import os

mediapipe_options = dict(
    static_image_mode=False,  # use history
    model_complexity=2,  # 0 - low, 2 - the biggest
    enable_segmentation=False,  # return additional output value - segmentation mask
    min_detection_confidence=0.5,  # detection threshold
    min_tracking_confidence=0.5,  # tracking threshold
)

webcam_options = {
    "fps": os.environ.get("fps", 30),
    "width": os.environ.get("width", 1920),
    "height": os.environ.get("height", 1080)
}

BG_COLOR = (192, 192, 192)  # gray
