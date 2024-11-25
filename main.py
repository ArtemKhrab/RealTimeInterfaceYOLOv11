from services.detector_service import DetectorService

if __name__ == "__main__":
    DetectorService(
        model_path="models/yolov11s_1class.pt"
    ).read_image(
        image_path="data/images.jpeg",
        show=True
    )
