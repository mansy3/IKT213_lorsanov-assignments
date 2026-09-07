import cv2
import numpy as np


def harris_corner_detection(reference_image):
    gray_image = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY
    )

    gray_image = np.float32(gray_image)

    harris = cv2.cornerHarris(
        gray_image,
        2,
        3,
        0.04
    )

    harris = cv2.dilate(
        harris,
        None
    )

    reference_image[
        harris > 0.01 * harris.max()
        ] = [0, 0, 255]

    return reference_image


def align_images(
        image_to_align,
        reference_image,
        max_features,
        good_match_precent
):
    gray_align = cv2.cvtColor(
        image_to_align,
        cv2.COLOR_BGR2GRAY
    )

    gray_reference = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect SIFT features
    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(
        gray_align,
        None
    )

    keypoints2, descriptors2 = sift.detectAndCompute(
        gray_reference,
        None
    )

    # FLANN matcher
    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    matches = flann.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    # Lowe's ratio test
    good_matches = []

    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good_matches.append(m)

    print("Good matches found:", len(good_matches))

    # max_features is used as minimum number of good matches
    if len(good_matches) < max_features:
        print(
            "Not enough matches are found -",
            len(good_matches),
            "/",
            max_features
        )
        return None, None

    source_points = np.float32(
        [
            keypoints1[m.queryIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    destination_points = np.float32(
        [
            keypoints2[m.trainIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0
    )

    height, width = reference_image.shape[:2]

    aligned_image = cv2.warpPerspective(
        image_to_align,
        homography,
        (width, height)
    )

    matches_mask = mask.ravel().tolist()

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask,
        flags=2
    )

    matches_image = cv2.drawMatches(
        image_to_align,
        keypoints1,
        reference_image,
        keypoints2,
        good_matches,
        None,
        **draw_params
    )

    return aligned_image, matches_image


def main():
    reference_image = cv2.imread(
        "reference_img.png"
    )

    image_to_align = cv2.imread(
        "align_this.jpg"
    )

    if reference_image is None:
        print("Could not load reference_img.png")
        return

    if image_to_align is None:
        print("Could not load align_this.jpg")
        return

    # Harris Corner Detection
    harris_image = harris_corner_detection(
        reference_image.copy()
    )

    cv2.imwrite(
        "images/harris.png",
        harris_image
    )

    print("Harris image saved.")

    # SIFT + FLANN alignment
    aligned_image, matches_image = align_images(
        image_to_align,
        reference_image,
        10,
        0.7
    )

    if aligned_image is not None:
        cv2.imwrite(
            "images/aligned.png",
            aligned_image
        )

        print("Aligned image saved.")

    if matches_image is not None:
        cv2.imwrite(
            "images/matches.png",
            matches_image
        )

        print("Matches image saved.")


if __name__ == "__main__":
    main()