import cv2
from supervision import Detections
from ultralytics import YOLO
import supervision
from services.annotator_service import AnnotatorService
from services.videostream_capture_service import VideostreamCaptureService
from services.visualize_service import VisualizeService


class DetectorService:
    """
    Service for performing object detection using YOLO models on different input sources.

    This service provides methods for real-time detection from webcam,
    video files, and static images using YOLO models.

    Attributes:
       CONFIDENCE_THRESHOLD (float): Minimum confidence threshold for detections (70%)
       model: YOLO model instance for object detection
    """
    CONFIDENCE_THRESHOLD = 70

    def __init__(self, model_path):
        """
        Initializes the DetectorService with the specified model.

        Args:
           model_path (str): Name or path of the YOLO model to be used for object detection.
        """
        self.model = YOLO(model=model_path, task="detect")

    def read_webcam(self, resolution: list = (1280, 720)) -> None:
        """
        Process webcam feed for real-time object detection and visualization.

        This method captures video from webcam, applies object detection model,
        filters detections by confidence, and displays the annotated results.

        Args:
           resolution: List of [width, height] for capture resolution,
                      defaults to [1280, 720]

        Note:
           - Press ESC (key code 27) to stop the video stream
           - Each frame goes through following pipeline:
             1. Capture from webcam
             2. Apply detection model
             3. Filter detections
             4. Annotate frame
             5. Display results

        """
        annotator_service = AnnotatorService()
        videostream_capture_service = VideostreamCaptureService(resolution=resolution)
        visualize_service = VisualizeService()
        while True:
            captured_frame = videostream_capture_service.read_video_stream()
            model_frame = self.model(captured_frame)[0]
            detections = supervision.Detections.from_ultralytics(model_frame)

            filtered_detections = self._filter_by_confidence(detections)
            annotated_frame = annotator_service.annotate(frame=captured_frame, detections=filtered_detections)

            visualize_service.visualize(frame=annotated_frame)

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

    def _filter_by_confidence(self, detections: Detections) -> Detections:
        """
        Filter detections based on confidence threshold.

        Args:
            detections: Detections object from YOLO model

        Returns:
            Detections: Filtered detections where confidence > threshold

        """
        filtered_detections = detections[detections.confidence > self.CONFIDENCE_THRESHOLD]
        return filtered_detections
