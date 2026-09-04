import cv2
import numpy as np


def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )

    return padded_image


def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]

    return cropped_image


def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))

    return resized_image


def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                emptyPictureArray[y, x, channel] = image[y, x, channel]

    return emptyPictureArray


def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image


def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return hsv_image


def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                new_value = int(image[y, x, channel]) + hue

                if new_value > 255:
                    new_value = new_value - 256

                if new_value < 0:
                    new_value = new_value + 256

                emptyPictureArray[y, x, channel] = new_value

    return emptyPictureArray


def smoothing(image):
    smoothed_image = cv2.GaussianBlur(
        image,
        (15, 15),
        0,
        borderType=cv2.BORDER_DEFAULT
    )

    return smoothed_image


def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

    return image


def main():
    image = cv2.imread("iris.png")

    if image is None:
        print("Could not load iris.png")
        return

    height, width, channels = image.shape

    # 1. Padding
    padded_image = padding(image, 100)
    cv2.imwrite("images/padding.png", padded_image)

    # 2. Crop
    cropped_image = crop(
        image,
        200,
        width - 130,
        200,
        height - 130
    )
    cv2.imwrite("images/crop.png", cropped_image)

    # 3. Resize
    resized_image = resize(image, 200, 200)
    cv2.imwrite("images/resize.png", resized_image)

    # 4. Manual copy
    emptyPictureArray = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    copied_image = copy(image, emptyPictureArray)
    cv2.imwrite("images/copy.png", copied_image)

    # 5. Grayscale
    grayscale_image = grayscale(image)
    cv2.imwrite("images/grayscale.png", grayscale_image)

    # 6. HSV
    hsv_image = hsv(image)
    cv2.imwrite("images/hsv.png", hsv_image)

    # 7. Hue shift
    emptyPictureArray = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    hue_image = hue_shifted(
        image,
        emptyPictureArray,
        50
    )
    cv2.imwrite("images/hue_shifted.png", hue_image)

    # 8. Smoothing
    smoothed_image = smoothing(image)
    cv2.imwrite("images/smoothing.png", smoothed_image)

    # 9. Rotation
    rotated_90 = rotation(image, 90)
    cv2.imwrite("images/rotation_90.png", rotated_90)

    rotated_180 = rotation(image, 180)
    cv2.imwrite("images/rotation_180.png", rotated_180)

    print("All images successfully created.")


if __name__ == "__main__":
    main()