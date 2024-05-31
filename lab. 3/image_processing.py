import numpy as np
import cv2 as cv
import networkx as nx
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from dsepruning import skel_pruning_DSE


MAX_BRIGHTNESS = 255
N_CLASSES = 4
CLASSES_VECTORS = [
    np.array([3, 1, 3, 3]),
    np.array([4, 0, 4, 1]),
    np.array([4, 0, 5, 2, 1]),
    np.array([6, 0, 4, 2])
]
CLASSES_REPR = ["I", "II", "III", "IV"]


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


class EquivalenceNodes:
    def __init__(self, nodes=None, thresh=25):
        self.thresh = thresh
        self.nodes = []
        self.centers_sum = np.array([0, 0])
        self.degrees_sum = 0
        if nodes is not None:
            for node in nodes:
                self.add(node)

    def get_mean_center(self):
        if len(self.nodes) == 0:
            return None
        return self.centers_sum / len(self.nodes)
    
    mean_center = property(get_mean_center)

    @staticmethod
    def _squared_dist(p0, p1):
        return (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
    
    def __contains__(self, node):
        return self._squared_dist(node["pts"][0], self.mean_center) < self.thresh ** 2

    def add(self, node):
        self.nodes.append(node)
        self.centers_sum += node["pts"][0]
        self.degrees_sum += node["degree"]

    def get_generalized_node(self):
        node = {
            "pts": np.array([self.mean_center], dtype=np.int16),
            "o": np.array(self.mean_center, dtype=np.uint16),
            "degree": self.degrees_sum - 2 * (len(self.nodes) - 1)
        }
        return node
    
    generalized_node = property(get_generalized_node)


def get_unique_nodes(graph, thresh=25):
    nodes_with_degrees = [(n, d) for n, d in graph.degree()]
    nodes = graph.nodes()

    equivalence_classes = []
    for node, degree in nodes_with_degrees:
        node = nodes[node].copy()
        node["degree"] = degree
        exists = False
        for equivalence_class in equivalence_classes:
            if node in equivalence_class:
                exists = True
                equivalence_class.add(node)
                break
        if not exists:
            equivalence_classes.append(EquivalenceNodes([node], thresh=thresh))

    unique_nodes = []
    for equivalence_class in equivalence_classes:
        unique_nodes.append(equivalence_class.generalized_node)

    return unique_nodes


def get_skel_and_graph(black_white_img, dse_min_area_px):
    skel = skeletonize(MAX_BRIGHTNESS - black_white_img)
    dist = distance_transform_edt(MAX_BRIGHTNESS - black_white_img, return_indices=False, return_distances=True)
    skel, graph = skel_pruning_DSE(skel, dist, min_area_px=dse_min_area_px, return_graph=True)
    return skel, graph


def get_img_with_skel(
        black_white_img, skel, graph, min_dist_thresh,
        skel_color=(0, 255, 0), skel_thikness=2,
        cicrles_color=(255, 0, 0), cicrles_center_thikness=5,
        cicrles_thikness=2,
        text_color=(255, 0, 255), text_thikness=2, text_scale=1,
        return_nodes=True
):
    img_with_skel =  black_white_img.copy()
    img_with_skel = cv.cvtColor(img_with_skel, cv.COLOR_GRAY2RGB)

    # draw edges by pts
    for (s, e) in graph.edges():
        for elem in graph[s][e].values():
            ps = elem['pts']
            pts = ps[:, ::-1].reshape((-1, 1, 2)).astype(np.int32)
            cv.polylines(img_with_skel, [pts], False, skel_color, skel_thikness)

    # draw node by o
    nodes = get_unique_nodes(graph, thresh=min_dist_thresh)
    for node in nodes:
        center = node["o"][::-1]
        # circle center
        cv.circle(img_with_skel, center, 1, cicrles_color, cicrles_center_thikness)
        # circle outline
        cv.circle(img_with_skel, center, min_dist_thresh, cicrles_color, cicrles_thikness)

        cv.putText(img_with_skel, str(node["degree"]), org=center + np.array([-3, 3]), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=text_scale, color=text_color, thickness=text_thikness, lineType=cv.LINE_AA)
    
    if return_nodes:
        return img_with_skel, nodes

    return img_with_skel


def _dist(vec1, vec2):
    l1 = vec1.shape[0]
    l2 = vec2.shape[0]
    if l1 < l2:
        vec1 = np.concatenate([vec1, np.zeros(l2 - l1, dtype="int")])
    elif l1 > l2:
        vec2 = np.concatenate([vec2, np.zeros(l1 - l2, dtype="int")])
    return np.sum((vec1 - vec2) ** 2)


def get_degree_vector_and_classify(nodes):
    degrees = sorted([node["degree"] for node in nodes])
    unique_degrees, counts_degrees = np.unique(degrees, return_counts=True)
    degree_vector = np.zeros(np.max(unique_degrees), dtype="int")

    for degree, count in zip(unique_degrees, counts_degrees):
        degree_vector[degree - 1] = count

    dists = np.zeros(N_CLASSES, dtype="int")
    for i, vec in enumerate(CLASSES_VECTORS):
        dists[i] = _dist(vec, degree_vector)

    min_dist_idx = np.argmin(dists)
    cls = CLASSES_REPR[min_dist_idx]

    return degree_vector, cls


    

