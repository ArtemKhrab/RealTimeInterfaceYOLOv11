import cv2
import supervision
from supervision import Detections


class AnnotatorService:
    """
    Service for annotating detected objects on frames with boxes and labels.

    This class initializes corner box and label annotators from the supervision library
    and provides methods to apply these annotations to frames with detected objects.

    Attributes:
       box_annotator: Supervision BoxCornerAnnotator for drawing corner boxes
       label_annotator: Supervision LabelAnnotator for adding labels with confidence scores
    """
    def __init__(self):
        """
        Initialize box and label annotators with default settings.

        Box annotator uses default corner style
        Label annotator uses red color for better visibility
        """
        self.box_annotator = supervision.BoxCornerAnnotator()
        self.label_annotator = supervision.LabelAnnotator(text_color=supervision.Color.RED)

    def annotate(self, frame: cv2.Mat, detections: Detections) -> cv2.Mat:
        """
        Apply box and label annotations to a frame with detections.

        Args:
           frame: OpenCV image matrix to annotate
           detections: Detected objects from YOLO model

        Returns:
           cv2.Mat: Annotated frame with boxes and labels
        """
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections)
        return frame
