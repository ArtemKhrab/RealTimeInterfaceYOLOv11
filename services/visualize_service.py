import cv2


class VisualizeService:
    """
    Service class for visualizing video frames.

    This class provides methods for displaying frames using OpenCV's system.
    """
    def visualize(self, frame: cv2.Mat) -> None:
        """
       Display a frame in a window.

       Args:
           frame: OpenCV image/frame to be displayed

       Note:
           Creates/updates a window named "OBJECT_DETECTOR" showing the frame.
           Use cv2.waitKey() after calling this method to properly display the window.
       """
        cv2.imshow("OBJECT_DETECTOR", frame)
