import cv2


def print_image_information(image):
    print("height:", image.shape[0])
    print("width:", image.shape[1])
    print("channels:", image.shape[2])
    print("size:", image.size)
    print("data type:", image.dtype)


def main():
    image = cv2.imread("iris-1.jpg")

    print_image_information(image)


if __name__ == "__main__":
    main()