import numpy as np
import subprocess
from PIL import Image
import cv2 as cv
import os
import warnings


MAX_BRIGHTNESS = 255


def binarize_thresh(gray_img, thresh=175, inverse=False):
    if inverse:
        gray_img = MAX_BRIGHTNESS - gray_img
        thresh = MAX_BRIGHTNESS - thresh
    return cv.threshold(gray_img, thresh, MAX_BRIGHTNESS, cv.THRESH_BINARY)[1]


def binarize_magic(color_img):
    """
    color_img: rgb format
    """
    r, g, b = [color_img[:, :, i] for i in range(3)]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    mask = (brightness > 200) & (r < b) | (r + 10 < b)

    black_and_white_img = np.full(color_img.shape[:2], MAX_BRIGHTNESS, dtype=np.uint8)
    black_and_white_img[mask] = 0

    return black_and_white_img


def make_morph_ex(black_and_white_img, ex, kernel_shape, object_is_black=True):
    if object_is_black:
        black_and_white_img = MAX_BRIGHTNESS - black_and_white_img

    try:
        iter(kernel_shape)
        kernel = np.ones(kernel_shape, np.uint8)
    except TypeError:
        kernel = np.ones((kernel_shape, kernel_shape), np.uint8)

    if ex == "erosion":
        res = cv.erode(black_and_white_img, kernel, iterations=1)
    elif ex == "dilation":
        res = cv.dilate(black_and_white_img, kernel, iterations=1)
    elif ex == "opening":
        res = cv.morphologyEx(black_and_white_img, cv.MORPH_OPEN, kernel)
    elif ex == "closing":
        res = cv.morphologyEx(black_and_white_img, cv.MORPH_CLOSE, kernel)
    else:
        raise NotImplementedError("unknown morph ex")
    
    if object_is_black:
        res = MAX_BRIGHTNESS - res

    return res


def get_img_with_contours_and_cnt(
        black_and_white_image, min_square=70,
        img_name="users_planes", thickness=2, color=(0, 0, 255),
        n_tries=5):
    
    if black_and_white_image is None:
        black_and_white_image = cv.imread(f"etc/{img_name}.bmp", cv.IMREAD_GRAYSCALE)
    else:
        img = Image.fromarray(black_and_white_image).convert("1")
        img.save(f"etc/{img_name}.bmp")

    img_with_contours = black_and_white_image.copy()
    img_with_contours = cv.cvtColor(
        img_with_contours, cv.COLOR_GRAY2RGB
    )

    max_y = img_with_contours.shape[0]

    for i in range(n_tries):
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.check_call([
                "etc/MedialRep_Server_v2.exe",
                f"etc/{img_name}.bmp",
                "2", # параметр стрижки скелета - минимальная длина ребра в пикселях (default=2)
                str(min_square) # минимальная площадь связной компоненты (default=10)
            ], startupinfo=si)
            break
        except subprocess.CalledProcessError:
            warnings.warn(f"MedialRep_Server_v2.exe crashed, try: {i}")

    with open(f"etc/{img_name}.txt", "r") as f:
        n_figures = int(f.readline().split()[3])
        f.readline() # skip n_outlines

        for _ in range(n_figures):
            f.readline()  # skip line "Figure {fig}"
            n_holes = int(f.readline().split()[3])
            f.readline()

            for _ in range(n_holes + 1):
                t = f.readline().split()[0]  # polygon type
                vert_num = int(f.readline().split()[3])

                if t == "External" and black_and_white_image is not None:
                    coords = np.zeros((vert_num, 2))
                    for i in range(vert_num):
                        x, y = map(int, f.readline().split()[1:])
                        coords[i, 0], coords[i, 1] = x, max_y - y
                    
                    img_with_contours = cv.polylines(
                        img_with_contours,
                        [coords.reshape((-1, 1, 2)).astype(np.int32)],
                        isClosed=False,
                        color=color,
                        thickness=thickness
                    )
                else:
                    # skip internal polygons
                    for _ in range(vert_num):
                        f.readline()

            f.readline()  # skip line "Skeleton of figure {fig}"
            n_nodes_sk = int(f.readline().split()[3])
            for _ in range(n_nodes_sk):
                f.readline() # skip skeleton nodes

            n_edges = int(f.readline().split()[3])
            f.readline()  # skip line "Number of controls {...}"
            for _ in range(n_edges):
                f.readline() # skip skeleton edges

    os.remove(f"etc/{img_name}.bmp")
    os.remove(f"etc/{img_name}.txt")
    
    return img_with_contours, n_figures
