from typing import Tuple

import cv2
import numpy as np
from numpy import typing as np_typing
from numpy import linspace as np_linspace
# from matplotlib import pyplot as plt


def load_image(image_1_path) -> np_typing.NDArray:
    return cv2.imread(image_1_path, cv2.COLOR_BGR2RGB)


def save_img(image_path: str, image: np_typing.NDArray):
    cv2.imwrite(image_path, image)


def resize_2_images(image_source_1, image_source_2) -> Tuple[np_typing.NDArray, np_typing.NDArray]:
    # resize images
    image_1 = image_source_1.copy()
    image_2 = image_source_2.copy()

    x1, y1, _ = image_1.shape
    x2, y2, _ = image_2.shape

    x = max([x1, x2])
    y = max([y1, y2])

    image_1 = cv2.resize(image_1, (y, x))
    image_2 = cv2.resize(image_2, (y, x))

    if _is_too_small(image_1) or _is_too_small(image_2):
        return image_source_1, image_source_2
    return image_1, image_2


def resize_3_images(image_source_1, image_source_2, image_source_3) -> Tuple[np_typing.ArrayLike, np_typing.ArrayLike, np_typing.ArrayLike]:
    # resize images
    image_1 = image_source_1.copy()
    image_2 = image_source_2.copy()
    image_3 = image_source_3.copy()

    x1, y1, _ = image_1.shape
    x2, y2, _ = image_2.shape
    x3, y3, _ = image_3.shape

    x = max([x1, x2, x3])
    y = max([y1, y2, y3])

    image_1 = cv2.resize(image_1, (y, x))
    image_2 = cv2.resize(image_2, (y, x))
    image_3 = cv2.resize(image_3, (y, x))

    if _is_too_small(image_1) or _is_too_small(image_2) or _is_too_small(image_3):
        return image_source_1, image_source_2, image_source_3
    return image_1, image_2, image_3


def resize_images_height(image_source_1, image_source_2):
    # resize images to the same height
    image_1 = image_source_1.copy()
    image_2 = image_source_2.copy()
    x1, y1, _ = image_1.shape
    x2, y2, _ = image_2.shape
    x = x2 if x1 < x2 else x1
    y1 = int(round(y1 * (x / x1)))
    y2 = int(round(y2 * (x / x2)))
    image_1 = cv2.resize(image_1, (y1, x))
    image_2 = cv2.resize(image_2, (y2, x))
    if _is_too_small(image_1) or _is_too_small(image_2):
        return image_source_1, image_source_2

    return image_1, image_2


def resize_images_width(image_source_1, image_source_2):
    # resize images to the same width
    image_1 = image_source_1.copy()
    image_2 = image_source_2.copy()
    x1, y1, _ = image_1.shape
    x2, y2, _ = image_2.shape
    y = y2 if y1 < y2 else y1
    x1 = int(round(x1 * (y / y1)))
    x2 = int(round(x2 * (y / y2)))
    image_1 = cv2.resize(image_1, (x1, y))
    image_2 = cv2.resize(image_2, (x2, y))
    if _is_too_small(image_1) or _is_too_small(image_2):
        return image_source_1, image_source_2
    return image_1, image_2


def scale_images(image_1, image_2):
    x1, y1, z1 = image_1.shape
    x2, y2, z2 = image_2.shape

    if x1 > x2 and y1 > y2:
        try:
            proportionX = x1 / x2
            proportionY = y1 / y2
            proportionX = proportionX if proportionX > 0 else 1
            proportionY = proportionY if proportionY > 0 else 1
            scaler = proportionX / proportionY if proportionX > proportionY else proportionY / proportionX
            x = round(x2 * scaler)
            y = round(y2 * scaler)
            image_2 = cv2.resize(image_2, (int(y), int(x)))
        except ZeroDivisionError:
            return image_1, image_2
    elif x2 > x1 and y2 > y1:
        try:
            proportionX = x2 / x1
            proportionY = y2 / y1
            proportionX = proportionX if proportionX > 0 else 1
            proportionY = proportionY if proportionY > 0 else 1
            scaler = proportionX / proportionY if proportionX > proportionY else proportionY / proportionX
            x = round(x1 * scaler)
            y = round(y1 * scaler)
            image_1 = cv2.resize(image_1, (int(y), int(x)))
        except ZeroDivisionError:
            return image_1, image_2
    return image_1, image_2


def crop_image(image_1, image_2):
    """ Make two image the same by width """
    image_1_crop = image_1.copy()
    image_2_crop = image_2.copy()
    x1, y1, z1 = image_1_crop.shape
    x2, y2, z2 = image_2_crop.shape
    sub1 = y1 / x1
    sub2 = y2 / x2
    if sub1 <= 0 or sub2 <= 0 or sub1 == sub2:
        return image_1, image_2

    if sub1 < sub2:
        y_crop = round((y2 - x2 * sub1) // 2)
        image_2_crop = image_2_crop[:, y_crop: -y_crop]
    elif sub2 < sub1:
        y_crop = round((y1 - x1 * sub2) // 2)
        image_1_crop = image_1_crop[:, y_crop: -y_crop]

    x1, y1, z1 = image_1.shape
    x2, y2, z2 = image_2.shape
    if (elem == 0 for elem in [x1, x2, y1, y2]):
        return image_1, image_2
    return image_1_crop, image_2_crop


def get_frames(image: 'np_typing.NDArray'):
    row1_, col1_, _ = image.shape
    img_copy = image.copy()
    subtractor = 5
    row1 = row1_ // subtractor
    col1 = col1_ // subtractor
    down_row1 = row1_ // subtractor
    up = img_copy[0: row1, :]
    left = img_copy[:, 0: col1]
    right = img_copy[:, -col1:]
    down = img_copy[-down_row1:, :]

    up_n_down = np.concatenate((up, down), axis=0)
    left_n_right = np.concatenate((left, right), axis=1)
    left_n_right_height, left_n_right_width, _ = left_n_right.shape
    up_n_down_height, up_n_down_width, _ = up_n_down.shape

    # Calculate the shape of the new array
    new_height = left_n_right_height + up_n_down_height
    new_width = left_n_right_width + up_n_down_width

    # Create a new numpy array with the desired shape
    result = np.zeros((new_height, new_width, 3))

    # Assign values from the original arrays to the new array
    result[:left_n_right_height, :left_n_right_width, :] = left_n_right
    result[left_n_right_height:, left_n_right_width:, :] = up_n_down
    return result


def crop_black_border(img):
    x_shape = img.shape[0] // 2
    y_shape = img.shape[1] // 2
    # top border
    top = img[: x_shape, : y_shape]
    gray_top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_top, 0, 255, cv2.THRESH_BINARY)
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    xend, yend, width, height = cv2.boundingRect(cnt)

    # down border
    down = img[x_shape:, y_shape:]
    rows, cols, _ = down.shape
    # rotate down to get border
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    dst = cv2.warpAffine(down, matrix, (cols, rows))
    gray_down = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh2 = cv2.threshold(gray_down, 0, 255, cv2.THRESH_BINARY)
    im, contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt2 = contours2[0]
    xend2, yend2, width2, height2 = cv2.boundingRect(cnt2)

    # if yend > 50 and yend2 > 50: # and all((xend < yend, xend2 < yend2)):
    crop = img[yend: -yend2, :]
    if _is_too_small(crop):
        return img
    return crop
    # return img


def get_template_matching(source, photo_to_compare, test_show=False):
    """Get the most similar part of the template in user photo (source)
    photo_to_compare - photo to compare
    source - user photo
    :return: matching pattern in users photo
    """
    photo = photo_to_compare.copy()
    source_ = source.copy()

    if any((source.shape[0] < photo_to_compare.shape[0],
            source.shape[1] < photo_to_compare.shape[1])):
        source_ = photo_to_compare.copy()
        photo = source.copy()

    def cut_edges_of_img():
        # cut some edges to make source smaller than template
        _shape_y = photo.shape[0] // 15
        _shape_x = photo.shape[1] // 15
        return photo[_shape_y: -_shape_y, _shape_x: -_shape_x]

    photo_ = cut_edges_of_img()

    # Apply template Matching and resize to find the best match
    # ----------------------
    image_matched, max_val_1 = _find_best_scale(source_, photo_)
    # ----------------------
    # find the same in source
    image_matched, source_ = scale_images(image_matched, source_)
    matched_in_photo, max_val_2 = _find_best_scale(photo, image_matched)
    # ----------------------
    return image_matched, matched_in_photo, max_val_1, max_val_2


def get_best_match(image_source1, image_source2):
    """ Iterate through two images, get best matches.
    Makes diff variation of scale to find best match.
    """
    img_original1, img_original2 = image_source1.copy(), image_source2.copy()

    # case 1
    img_resized_1, img_resized_2 = \
        resize_2_images(img_original1, img_original2)
    img_resized1, img_resized2, max_val_1, max_val_2 = get_template_matching(
        img_resized_1.copy(), img_resized_2.copy()
    )
    max_val_resized = max_val_2 if max_val_2 > max_val_1 else max_val_1
    # not correct result, extract .1 from final max val
    max_val_resized = max_val_resized - 0.1 if max_val_resized > 0 else 0

    # Prepare for next cases
    img_scaled1, img_scaled2 = \
        scale_images(img_original1, img_original2)

    # case 2
    try:
        img_cropped_resized_height1, img_cropped_resized_height2 = \
            resize_images_height(img_scaled1, img_scaled2)
        img_cropped_resized_height1, img_cropped_resized_height2, max_val_1, max_val_2 = \
            get_template_matching(img_cropped_resized_height1,
                                       img_cropped_resized_height2)
        max_val_resized_height = max_val_2 if max_val_2 > max_val_1 else max_val_1
    except cv2.error:
        max_val_resized_height = 0
    # case 3
    try:
        img_cropped_resized_width1, img_cropped_resized_width2 = \
            resize_images_width(img_scaled1, img_scaled2)
        img_cropped_resized_width1, img_cropped_resized_width2, max_val_1, max_val_2 = \
            get_template_matching(img_cropped_resized_width1,
                                       img_cropped_resized_width2)
        max_val_resized_width = max_val_2 if max_val_2 > max_val_1 else max_val_1
    except cv2.error:
        max_val_resized_width = 0
    # case 4
    try:
        img_cropped_resized1, img_cropped_resized2 = \
            resize_2_images(img_scaled1, img_scaled2)
        img_cropped_resized1, img_cropped_resized2, max_val_1, max_val_2 = \
            get_template_matching(img_cropped_resized1,
                                       img_cropped_resized2)
        max_val_cropped_scaled = max_val_2 if max_val_2 > max_val_1 else max_val_1
    except cv2.error:
        max_val_cropped_scaled = 0
    # ----- Find best match ---------
    max_final_val = max((max_val_resized,
                         max_val_resized_height,
                         max_val_resized_width,
                         max_val_cropped_scaled
                         ))

    switch = {
        max_val_resized: (img_resized1, img_resized2),
        max_val_resized_height:
            (img_cropped_resized_height1, img_cropped_resized_height2),
        max_val_resized_width: (img_cropped_resized_width1, img_cropped_resized_width2),
        max_val_cropped_scaled: (img_cropped_resized1, img_cropped_resized2)
    }

    image_1, image_2 = switch.get(max_final_val, (img_original1, img_original2))
    if any((image_1.shape[0] < 80, image_1.shape[1] < 80, image_2.shape[0] < 80, image_2.shape[1] < 80)):
        return img_resized_1, img_resized_2
    return image_1, image_2


def _find_best_scale(source_img, templ_img):
    """Scale template to find best match.
    Return best matched image and max Value of match
    """
    source = source_img.copy()
    template = templ_img.copy()
    try:
        gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray_source = source

    (tH, tW) = template.shape[:2]
    try:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray_template = template

    image_matched = None
    found = None
    # loop over the scales of the image
    # can't use negative step ValueError: Number of samples, -20, must be non-negative.
    # [::-1] for backwards iteration
    for scale in np_linspace(0.8, 1.0, 10)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        x1 = gray_template.shape[0]
        y1 = gray_template.shape[1]
        x = round(x1 * scale)
        y = round(y1 * scale)
        resized = cv2.resize(gray_template, (int(y), int(x)))

        # if the resized image is smaller than the template, then break from the loop
        if any((source.shape[0] < resized.shape[0],
                source.shape[1] < resized.shape[1],
                resized.shape[0] < 10)):
            break

        # matching to find the template in the image
        result = cv2.matchTemplate(gray_source, resized, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, scale)

    if found is None:
        return source_img, 0

    maxVal, top_of_result, scale = found
    startX, startY = int(top_of_result[0] * scale), int(top_of_result[1] * scale)
    bottom_of_res = (startX + tW, startY + tH)
    endX, endY = bottom_of_res[0], bottom_of_res[1]

    image_matched = source_img[startX: endX, startY: endY]
    return image_matched, maxVal

# def show_images(im1, im2, title1, title2):
#     plt.subplot(121), plt.imshow(im1)
#     plt.title(title1), plt.xticks([]), plt.yticks([])
#
#     plt.subplot(122), plt.imshow(im2)
#     plt.title(title2), plt.xticks([]), plt.yticks([])
#     plt.show()


def _is_too_small(image):
    if image.shape[0] < 80 or image.shape[1] < 80:
        return True


def flip_img(image):
    return cv2.flip(image, 1)


def rotate(image: 'np_typing.NDArray', degree: int = 25) -> 'np_typing.NDArray':
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
