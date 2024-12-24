import cv2


class VideostreamCaptureService:
    """
    A service class for capturing video stream from a camera device.

    This class initializes a video capture device with specified resolution
    and provides methods to read frames from the video stream.

    Attributes:
        captured_source: OpenCV VideoCapture object
    """
    def __init__(self, resolution: list) -> None:
        """
        Initialize the video capture service with specified resolution.

        Args:
            resolution: A list of (width, height) specifying the desired frame resolution

        """
        frame_width, frame_height = resolution

        self.captured_source = cv2.VideoCapture(0)
        self.captured_source.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.captured_source.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    def read_video_stream(self) -> cv2.Mat:
        """
        Capture and return a single frame from the video stream.

        Returns:
            cv2.UMat: A captured frame from the video stream.
            If capture fails, returns None.

        """
        _, captured_frame = self.captured_source.read()
        return captured_frame
