import cv2
import numpy as np


def sobel_edge_detection(image):
    blurred_image = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    sobel_image = cv2.Sobel(
        blurred_image,
        cv2.CV_64F,
        dx=1,
        dy=1,
        ksize=1
    )

    sobel_image = cv2.convertScaleAbs(sobel_image)

    return sobel_image


def canny_edge_detection(image, threshold_1, threshold_2):
    blurred_image = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    canny_image = cv2.Canny(
        blurred_image,
        threshold_1,
        threshold_2
    )

    return canny_image


def template_match(image, template):
    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray_template = cv2.cvtColor(
        template,
        cv2.COLOR_BGR2GRAY
    )

    result = cv2.matchTemplate(
        gray_image,
        gray_template,
        cv2.TM_CCOEFF_NORMED
    )

    threshold = 0.9

    locations = np.where(
        result >= threshold
    )

    template_height, template_width = gray_template.shape

    for point in zip(*locations[::-1]):
        cv2.rectangle(
            image,
            point,
            (
                point[0] + template_width,
                point[1] + template_height
            ),
            (0, 0, 255),
            2
        )

    return image


def resize(image, scale_factor: int, up_or_down: str):
    height, width = image.shape[:2]

    if up_or_down == "up":
        resized_image = cv2.pyrUp(
            image,
            dstsize=(
                width * scale_factor,
                height * scale_factor
            )
        )

    elif up_or_down == "down":
        resized_image = cv2.pyrDown(
            image,
            dstsize=(
                width // scale_factor,
                height // scale_factor
            )
        )

    else:
        print("up_or_down must be 'up' or 'down'")
        return image

    return resized_image


def main():
    image = cv2.imread("lambo.png")

    if image is None:
        print("Could not load lambo.png")
        return

    # sobel edge detection
    sobel_image = sobel_edge_detection(image)

    cv2.imwrite(
        "images/sobel.png",
        sobel_image
    )

    print("Sobel image saved.")

    # canny edge detection
    canny_image = canny_edge_detection(
        image,
        50,
        50
    )

    cv2.imwrite(
        "images/canny.png",
        canny_image
    )

    print("Canny image saved.")

    # template matching
    shapes_image = cv2.imread("shapes.png")
    shapes_template = cv2.imread("shapes_template.jpg")

    if shapes_image is None:
        print("Could not load shapes.png")
        return

    if shapes_template is None:
        print("Could not load shapes_template.jpg")
        return

    template_match_image = template_match(
        shapes_image,
        shapes_template
    )

    cv2.imwrite(
        "images/template_match.png",
        template_match_image
    )

    print("Template match image saved.")

    # resize up
    resize_up_image = resize(
        image,
        2,
        "up"
    )

    cv2.imwrite(
        "images/resize_up.png",
        resize_up_image
    )

    print("Resize up image saved.")

    # resize down
    resize_down_image = resize(
        image,
        2,
        "down"
    )

    cv2.imwrite(
        "images/resize_down.png",
        resize_down_image
    )

    print("Resize down image saved.")


if __name__ == "__main__":
    main()