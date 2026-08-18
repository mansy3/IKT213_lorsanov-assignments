import cv2


def save_camera_information():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open camera")
        return

    fps = camera.get(cv2.CAP_PROP_FPS)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)

    with open("camera_outputs.txt", "w") as file:
        file.write(f"fps: {fps}\n")
        file.write(f"height: {height}\n")
        file.write(f"width: {width}\n")

    camera.release()


def main():
    save_camera_information()


if __name__ == "__main__":
    main()