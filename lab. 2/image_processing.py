import numpy as np
import cv2 as cv
from typing import List


def is_point_in_mask(x0, y0, mask, radius=2):
    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                continue
            if mask[y, x]:
                return True
            
    return False


def get_segs_borders(x0, y0, x1, y1, mask, min_seg_len=10, max_gap=5, radius=2, max_seg_len=100):
    """
    (x0, y0), (x1, y1) - задают прямую
    return: список из двух точек (p_0, p_1) - начало и конец отрезков
    """
    # алгоритм Брезенхэма

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx - dy

    seg_borders = []
    len_counter = 0
    gap_counter = 0
    while True:
        if is_point_in_mask(x0, y0, mask, radius):
            if len_counter == 0:
                p0_seg = (x0, y0)
            len_counter += 1 + gap_counter
            gap_counter = 0
        else:
            if len_counter != 0 and gap_counter < max_gap:
                gap_counter += 1
            else:
                if len_counter >= min_seg_len and len_counter <= max_seg_len:
                    p1_seg = (x0, y0)
                    seg_borders.append((p0_seg, p1_seg))
                len_counter = 0
                gap_counter = 0


        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return seg_borders


def squared_dist(p0, p1):
    return (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2


def dist_to_line(point, vert1, vert2):
    """
    Расстояние от точки point до прямой, заданной точками vert1 и vert2
    """
    return np.linalg.norm(np.cross(vert2 - vert1, vert1 - point)) / np.linalg.norm(vert2 - vert1)


def get_neigh(x0, y0, gray_image, radius):
    x_idx = np.arange(x0 - radius, x0 + radius + 1)
    y_idx = np.arange(y0 - radius, y0 + radius + 1)
    x_idx = x_idx[(x_idx >= 0) & (x_idx < gray_image.shape[1])]
    y_idx = y_idx[(y_idx >= 0) & (y_idx < gray_image.shape[0])]
    neigh = gray_image[y_idx, :][:, x_idx].reshape(-1)
    return neigh


def mean_brightness_neigh(x0, y0, gray_image, radius):
    neigh = get_neigh(x0, y0, gray_image, radius)
    return neigh.mean()


class Triangle:
    def __init__(self, p0, p1, gray_image, **kwargs):
        self.points = [np.array(p0), np.array(p1)]
        self.direction = self._get_direction(gray_image, **kwargs)
        self.center = None
        self._set_last_point_and_center()
        self.height = self._get_height()

    def _get_direction(self, gray_image, offset=15, radius=3):
        """
        возвращает положение треугольника относительно отрезка (стороны теугольника)
        p0, p1 - начало и конец отрезка
        image - исходное RGB изображение
        return: "top", или "bottom", или "left", или "right"
        """
        if offset <= radius:
            raise ValueError("offset <= radius")
        
        p0, p1 = self.points

        x_center = (p0[0] + p1[0]) // 2
        y_center = (p0[1] + p1[1]) // 2
        
        dx = abs(p0[0] - p1[0])
        dy = abs(p0[1] - p1[1])

        if dx > dy:
            y_bottom = y_center + offset
            y_top = y_center - offset
            top_br = mean_brightness_neigh(x_center, y_top, gray_image, radius)
            bottom_br = mean_brightness_neigh(x_center, y_bottom, gray_image, radius)
            if top_br < bottom_br:
                return "top"
            return "bottom"
        
        x_left = x_center - offset
        x_right = x_center + offset
        left_br = mean_brightness_neigh(x_left, y_center, gray_image, radius)
        right_br = mean_brightness_neigh(x_right, y_center, gray_image, radius)
        if left_br < right_br:
            return "left"
        return "right"


    def _set_last_point_and_center(self):
        p0 = self.points[0]
        p1 = self.points[1]

        if p1[1] < p0[1]:
            p0, p1 = p1, p0

        x_d, y_d = p0[0] - p1[0], p0[1] - p1[1]

        if p1[0] > p0[0]:
            theta = np.pi / 3
            if self.direction in ["top", "right"]:
                theta = -theta
        else:
            theta = -np.pi / 3
            if self.direction in ["top", "left"]:
                theta = -theta

        x_last = int(x_d * np.cos(theta) + y_d * np.sin(theta)) + p1[0]
        y_last = int(- x_d * np.sin(theta) + y_d * np.cos(theta)) + p1[1]

        self.points.append(np.array((x_last, y_last)))

        x_seg_1_center = (p0[0] + p1[0]) // 2
        y_seg_1_center = (p0[1] + p1[1]) // 2

        x_center = (x_last + x_seg_1_center * 2) // 3
        y_center = (y_last + y_seg_1_center * 2) // 3

        self.center = np.array((x_center, y_center))

    def _get_height(self):
        p0, p1, p2 = self.points
        other_side_center = (p1 + p2) / 2
        return np.linalg.norm(p0 - other_side_center)

    def get_size(self):
        return squared_dist(self.center, self.points[0])
    
    size = property(get_size)

    def draw(
        self,
        image,
        sides_color=(0, 0, 255),
        built_side_thickness=1,
        main_side_thickness=1,
        draw_center=True,
        center_color=(0, 0, 255),
        center_radius=3
    ):
        cv.line(image, self.points[0], self.points[1], sides_color, main_side_thickness)
        cv.line(image, self.points[1], self.points[2], sides_color, built_side_thickness)
        cv.line(image, self.points[0], self.points[2], sides_color, built_side_thickness)
        if draw_center:
            cv.circle(image, self.center, center_radius, center_color)

    def contains(self, point, indentation=10):
        """
        indentation: если точка расположена к какой-либо из вершин ближе, чем на indentation,
            то считается, что она не принадлежит треугольнику.
        return: tuple(bool, int or None) - содержит ли треугольник точку, если да,
            то также возвражается номер ближайшей вершины
        """
        dists = np.zeros(3)
        p0, p1, p2 = self.points

        for i, (vert1, vert2) in enumerate([(p1, p2), (p0, p2), (p0, p1)]):
            d = dist_to_line(point, vert1, vert2)
            if d <= indentation:
                return False, None
            dists[i] = d

        if dists.sum() > self.height + 1:
            return False, None

        return True, np.argmax(dists)

    def __contains__(self, point):
        return self.contains(point)[0]


class EquivalenceTriangles:
    def __init__(self, triangles=None):
        self.triangles = []
        self.centers_sum = np.array([0, 0])
        if triangles is not None:
            for triangle in triangles:
                self.add(triangle)

    def get_mean_center(self):
        if len(self.triangles) == 0:
            return None
        return self.centers_sum / len(self.triangles)
    
    mean_center = property(get_mean_center)

    def in_equivalence_class(self, triangle, thresh=30):
        return squared_dist(triangle.center, self.mean_center) <= thresh ** 2
    
    def __contains__(self, triangle):
        return self.in_equivalence_class(triangle)

    def add(self, triangle):
        self.triangles.append(triangle)
        self.centers_sum += triangle.center

    def get_generalized_triangle(self, idx=-1):
        sizes = np.zeros(len(self.triangles))
        for i, triangle in enumerate(self.triangles):
            sizes[i] = triangle.size

        idx = np.argsort(sizes).take(idx, mode="wrap")
        return self.triangles[idx]
    
    generalized_triangle = property(get_generalized_triangle)


def get_unique_triangles(triangles):
    equivalence_classes = []
    for triangle in triangles:
        exists = False
        for equivalence_class in equivalence_classes:
            if triangle in equivalence_class:
                exists = True
                equivalence_class.add(triangle)
                break
        if not exists:
            equivalence_classes.append(EquivalenceTriangles([triangle]))

    unique_triangles = []
    for equivalence_class in equivalence_classes:
        unique_triangles.append(equivalence_class.generalized_triangle)

    return unique_triangles


class ColorClf:
    def predict(self, x):
        r, g, b = x[0]
        brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
        if  brightness >= 140 or (brightness > 130 and r / b < 1.4 and
                                 g / b < 1.4 and b / g < 1.4 and b / r < 1.4):
            # white
            return 1
        elif (g > b and g + 5 > r and 20 < brightness and brightness < 80) or (
            30 < brightness and brightness < 50 and g < r and r / g < 1.3
            and b < g and g / b < 1.3):
            # green
            return 2
        elif (b < 40 and r > 130 and g > 80) or (r > 7 * b and g > 50):
            # yellow
            return 3
        elif b > g + 10 and b > r + 10 and brightness > 40:
            # blue
            return 4
        elif r > 2.7 * g:
            # red
            return 5
        else:
            # wood
            return 0


def classify(
        triangles: List[Triangle],
        circles: np.ndarray,
        image,
        clf,
        indentation=3,
        return_logits=False,
        radius=1,
        n_ignore_points=1,
        draw_circles=False,
        img_to_draw=None
    ):
    """
    triangles: сегментированные треугольники
    circles: найденные окружности
    image: RGB-изображение
    
    """
    if circles is None:
        raise ValueError("circles is None")
    
    rgb_comps = [image[:, :, i] for i in range(3)]
    
    colored_points_counters = np.zeros((len(triangles), 3, 5), dtype="int")
    for i in circles[0, :]:
        circ_center = (int(i[0]), int(i[1]))
        circ_radius = int(i[2])
        neighes = [get_neigh(circ_center[0], circ_center[1], comp, radius) for comp in rgb_comps]
        for triangle_idx, triangle in enumerate(triangles):
            contains, nearest = triangle.contains(circ_center, indentation=indentation)
            if not contains:
                continue
            for r, g, b in zip(*neighes):
                pred = clf.predict(np.array([[r, g, b]]))
                if pred != 0:
                    colored_points_counters[triangle_idx, nearest, pred - 1] += 1

            if draw_circles:
                if img_to_draw is None:
                    img_to_draw = image
                cv.circle(img_to_draw, circ_center, 1, (255, 0, 0), 1)
                cv.circle(img_to_draw, circ_center, circ_radius, (255, 0, 255), 1)

            break

    colored_points_counters[colored_points_counters <= n_ignore_points] = 0
    labels = np.zeros((len(triangles), 3), dtype="int")
    sums = np.sum(colored_points_counters, axis=2)
    labels[sums != 0] = np.argmax(colored_points_counters, axis=2)[sums != 0] + 1

    if return_logits:
        return labels, colored_points_counters

    return labels