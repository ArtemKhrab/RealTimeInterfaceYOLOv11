import cv2
import supervision
from ultralytics import YOLO


class DetectorService:
    """
    A service for object detection using YOLO models. This class supports detection
    in real-time webcam feeds, video files, and static images.

    Attributes:
        model (str): Name of the YOLO model to be used for detection.

    Methods:
        __init__(model_name):
            Initializes the DetectorService with the specified model.

        read_webcam() -> None:
            Starts real-time object detection using a webcam feed.

        read_video(video_path: str) -> None:
            Runs object detection on a video file specified by the path.

        read_image(image_path: str) -> None:
            Runs object detection on a static image specified by the path.
    """

    def __init__(self, model_path):
        """
        Initializes the DetectorService with the specified model.

        Args:
           model_path (str): Name or path of the YOLO model to be used for object detection.
        """
        self.model = YOLO(model=model_path, task="detect")

    def read_webcam(self, resolution: list = (1280, 720)) -> None:
        """
        Starts real-time object detection using a webcam feed.

        Args:
           resolution (list): List or tuple specifying the resolution of the webcam feed as (width, height).
                              Default is (1280, 720).

        This method initializes the webcam with the specified resolution, processes each frame
        using the YOLO model, and displays the detection results in a window. Press 'Esc' to terminate the detection.

        Raises:
           SystemExit: If there are errors with webcam access.
        """
        frame_width, frame_height = resolution

        captured_source = cv2.VideoCapture(0)
        captured_source.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        captured_source.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        box_annotator = supervision.BoxCornerAnnotator()
        label_annotator = supervision.LabelAnnotator(text_color=supervision.Color.RED)

        while True:
            _, captured_frame = captured_source.read()
            model_frame = self.model(captured_frame)[0]
            found_detections = supervision.Detections.from_ultralytics(model_frame)

            captured_frame = box_annotator.annotate(scene=captured_frame, detections=found_detections)
            captured_frame = label_annotator.annotate(scene=captured_frame, detections=found_detections)
            cv2.imshow("MIL_OBJECT_DETECTOR", captured_frame)

            if cv2.waitKey(30) == 27:
                break

    def read_video(self, video_path: str, show: bool = False) -> None:
        """
        Runs object detection on a video file.

        Args:
            video_path (str): Path to the video file for detection.
            show (bool): show result.

        Raises:
            FileNotFoundError: If the video file is not found at the specified path.
        """
        self.model(source=video_path, show=show)

    def read_image(self, image_path: str, show: bool = False) -> None:
        """
        Runs object detection on a static image.

        Args:
           image_path (str): Path to the image file for detection.
           show (bool): show result.

        Raises:
           FileNotFoundError: If the image file is not found at the specified path.
        """
        self.model(source=image_path, show=show)
