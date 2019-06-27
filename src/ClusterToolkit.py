import math
import operator
import itertools
from pprint import pprint
import cv2
import openslide
import numpy as np
from multiprocessing import Pool
from functools import partial
from sklearn.metrics.cluster import adjusted_rand_score


class ClusterToolkit:
    """Toolkit for analyzing a whole slide image using K-means classifiers"""

    def __init__(self, n_clusters=3):
        """ Initialization for clustering toolkit

        :param n_clusters: K clusters that you want to segment the image into, default is 3
        """

        self.n_clusters = n_clusters
        self.cluster_list = None
        self.images = None

    def get_images(self):
        return self.images

    def read_images(self, tiles):
        """ Takes list input of tiles, returns list of contained images opened using cv2

        :param tiles: list of "tiles" which contain coordinates and corresopnding images in the format [(x, y), Image]
        """

        print("[READING IMAGES]")

        images = [cv2.cvtColor(np.array(tile[1]), cv2.COLOR_BGR2RGB) for tile in tiles]
        self.images = [
            image.reshape((image.shape[0] * image.shape[1], 3)).astype(np.float32)
            for image in images
        ]

    def find_optimal_k(self, range_min=3, range_max=7, n_iterations=50, n_jobs=1):
        """ Uses images stored in self.images to find the optimal number of groupings of pixels for the whole slide

        First, this function iterates through each image, running a k-means classification for n interations per image
        and storing the classification labels in a prediction dictionary, key_list. The keys in key_list are tuples
        containing the index of the image and the current K.

        Then, it takes the RAND index between each classification and all other classifications per image per K,
        takes the mean of RAND indexes per image, and then takes the mean of average rand indexes across the images.

        The K with the most stable classifications across all images is selected, and saved to self.n_clusters

        :param range_min: Minimum number (inclusive) of centroids to test
        :param range_max: Maximum number (exclusive) of centroids to test
        :param n_iterations: Number of times to run K-means classification per image per K
        :param n_jobs: Number of processes to create
        :return: None- saves K with most stable classifications to self.n_clusters
        """

        print("[FINDING OPTIMAL K]")

        images = self.images
        # TODO: Lump together finding RAND and this part
        key_list = []
        for index, image in enumerate(images):  # Iterate through images
            for k in range(
                range_min, range_max
            ):  # Iterate through range of K's for that image
                key_list.append((index, k))

        print(len(key_list), "keys found")
        pool = Pool(processes=n_jobs)
        rand_partial = partial(
            find_stable_rand, image_list=self.images, iterations=n_iterations
        )
        image_rands = pool.map(rand_partial, key_list)

        pool.close()
        pool.join()
        image_rand_dict = {}
        for pair in image_rands:
            image_rand_dict[pair[0]] = pair[1]

        del image_rands

        self.cross_key_rands(image_rand_dict)

        print(f"Segmentation produces most stable results with k={self.n_clusters}")

    def cross_key_rands(self, rand_dict):
        k_rands = {}
        for key in rand_dict.keys():
            if key[1] in k_rands.keys():
                k_rands[key[1]].append(rand_dict[key])
            else:
                k_rands[key[1]] = []

        for k in k_rands:
            k_rands[k] = sum(k_rands[k]) / len(k_rands[k])

        pprint(k_rands)
        self.n_clusters = max(k_rands.items(), key=operator.itemgetter(1))[0]

    def find_clusters(self, cluster_stride=1):
        """ Iterates through a set of images with specified cluster stride, calculating K-means for each image

        :param cluster_stride: stride for iterating over images, increasing this speeds up the process but reduces
        and may overlook small but distinct regions
        :return: None- sets self.cluster_list to full list of the k centroids from the tiles
        """

        print("[CLASSIFYING TILES]")

        cluster_centers = None

        for image in self.images[::cluster_stride]:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(
                image, self.n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )

            if cluster_centers is None:
                cluster_centers = np.array([center.astype(int)])
            else:
                cluster_centers = np.append(
                    cluster_centers, [center.astype(int)], axis=0
                )

        self.cluster_list = cluster_centers.astype(np.float32)

    def supercluster(self):
        """ Runs k-means classifier on list of tile centroids stored in self.cluster_list

        :return: k-means classifier tuple containing (compactness, data labels, centroids)
        """

        print("[CALCULATING SUPERCLUSTER]")

        supercluster = None

        for cluster in self.cluster_list:
            if supercluster is None:
                supercluster = np.array(cluster)
            else:
                supercluster = np.append(supercluster, cluster, axis=0)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.00001)
        kmeans_classifier = cv2.kmeans(
            supercluster, self.n_clusters, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS
        )
        print("Cluster centers:")
        print(kmeans_classifier[2])
        return kmeans_classifier

    def write_cluster_file(self, output_path):
        """ Writes .txt file containing cluster centers- useful for testing and not recomputing tiling/kmeans

        :param output_path: path to write file to
        :return: None
        """

        print("[WRITING CLUSTER FILE]")

        with open(output_path, "w") as f:
            for cluster in self.cluster_list:
                f.write("|=nc=|\n")
                for centroid in cluster:
                    f.write("%s\n" % str(centroid)[1:-1])
                f.write("|=ec=|\n")

    def read_cluster_file(self, file_path):
        """Read in cluster file from write_cluster_file, sets self.cluster_list to list of all centroids

        :param file_path: path to written cluster file (.txt)
        :return: None- sets self.cluster_list to list of all centroids stored in file
        """

        print("[READING CLUSTER FILE]")

        in_cluster = False
        clusters = None
        current_cluster = None

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()

                # Find start of new cluster, start recording lines
                if "|=nc=|" in line.strip():
                    in_cluster = True
                    continue

                # If at end of cluster, add to cluster list
                elif "|=ec=|" in line.strip():
                    in_cluster = False
                    if clusters is None:
                        clusters = np.array(current_cluster)
                    else:
                        clusters = np.concatenate(
                            ([clusters], [current_cluster]), axis=0
                        )

                    current_cluster = None
                    continue

                # While in cluster, append lines to cluster
                if in_cluster:
                    centroid_vals = np.array([int(i) for i in line.split(" ")])

                    if current_cluster is None:
                        current_cluster = np.array([centroid_vals])
                    else:
                        current_cluster = np.concatenate(
                            (current_cluster, [centroid_vals]), axis=0
                        )

        self.cluster_list = clusters


def downscale_tile(tile):
    tile[1] = downscale(tile[1])
    return tile


def find_stable_rand(key, image_list, iterations):
    predictions = []
    image = image_list[key[0]]

    for i in range(iterations):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(
            image, key[1], None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS
        )
        predictions.append(label.flatten())

    rand_scores = []
    for a, b in itertools.combinations(predictions, 2):
        rand_scores.append(adjusted_rand_score(a, b))

    del predictions

    rand_scores = sum(rand_scores) / len(rand_scores)
    return key, rand_scores


def create_masks(tile_map, classifier, gaussian_kernel=9, njobs=1):
    """ Given a map of tiles containing (coordinates, image), draw classification masks of each tile

    :param tile_map: Map of tiles containing (coordinates, image)
    :param classifier: superclustered k-means classifier tuple containing (compactness, data labels, centroids)
    :param gaussian_kernel: kernel for gaussian blur to smooth results
    :param njobs: number of processors to use
    :return: mask_tiles- tuples containing ((coordinates), image mask)
    """

    print("[CREATING MASKS]")

    pool = Pool(processes=njobs)
    mask_part = partial(__draw_mask, classifier=classifier, kernel=gaussian_kernel)
    mask_tiles = pool.map(mask_part, tile_map)
    pool.close()
    pool.join()

    return mask_tiles


def __draw_mask(tile, classifier, kernel):
    """ Helper method to draw masks on a tile using a k-means classifier

    :param tile: tuple containing ((coordinates), image)
    :param classifier: k-means classifier tuple containing (compactness, data labels, centroids)
    :param kernel: kernel for gaussian bluring for smoother results
    :return: tuple containing ((coordinates), image mask)
    """

    blurred_image = cv2.GaussianBlur(
        cv2.cvtColor(tile[1], cv2.COLOR_BGR2RGB), (kernel, kernel), 0
    )

    coords = tile[0]
    colors = classifier[2]

    maximum = 0
    max_index = -1

    for index, color in enumerate(colors):
        if sum(color) / 3 > maximum:
            max_index = index

    colors[max_index] = np.array([255, 255, 255])  # Set brightest color to white

    prediction_matrix = np.zeros(
        (tile[1].shape[0], tile[1].shape[1], 3), dtype=np.uint8
    )

    for x in range(tile[1].shape[0]):
        for y in range(tile[1].shape[1]):
            prediction_matrix[x][y] = __cv2_predict(colors, blurred_image[x][y])

    return tuple([coords, prediction_matrix])


def __cv2_predict(cluster_centers, new_sample):
    """ Helper method which predicts which centroid a new sample is closest to.

    Works by computing Euclidean distance between cluster center channels and the sample channels

    :param cluster_centers: cluster centers for K-means classifier
    :param new_sample: Values of feature vector to predict classification for
    :return: RGB value of cluster center closest to the new sample
    """

    distances = {}
    for index, centroid in enumerate(
        cluster_centers
    ):  # Find Euclidean distance between new sample and each centroid
        sum_of_squares = 0
        for channel_index in range(len(centroid)):
            sum_of_squares += (centroid[channel_index] - new_sample[channel_index]) ** 2

        distances[index] = math.sqrt(sum_of_squares)

    return cluster_centers[
        min(distances.items(), key=operator.itemgetter(1))[0]
    ].astype(np.uint8)


def downscale(image, downscaling_factor=(16, 16)):
    """Helper method to downscale cv2 image for easier computation"""
    return cv2.resize(
        image, None, fx=1 / (downscaling_factor[1]), fy=1 / downscaling_factor[0]
    )


def detect_tissue(slide_path, tile_size, zoom=0, njobs=1):
    slide = openslide.open_slide(slide_path)
    thumb_width = int(slide.level_dimensions[zoom][0] / tile_size[0])
    thumb_height = int(slide.level_dimensions[zoom][1] / tile_size[1])
    thumbnail = np.array(slide.get_thumbnail((thumb_width, thumb_height)))
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
    # print(thumb_width, thumb_height)
    slide.close()

    # cv2.imshow("thumbnail", thumbnail)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    thumb_mask = np.zeros((thumbnail.shape[0], thumbnail.shape[1], 1))

    for x in range(thumbnail.shape[0]):
        for y in range(thumbnail.shape[1]):
            # print(thumbnail[x][y])
            if sum(thumbnail[x][y][:3] / 3) > 236:
                # print("nope")
                thumb_mask[x][y] = 0
            else:
                # print("yep")
                thumb_mask[x][y] = 1

    # print(thumb_mask.shape)
    # cv2.imshow("Masked", thumb_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for x in range(thumb_mask.shape[0]):
        for y in range(thumb_mask.shape[1]):
            surrounding_sum = 0

            if thumb_mask[x][y] == 0:  # if tile is empty, check surrounding 8 tiles
                if x - 1 >= 0:
                    surrounding_sum += thumb_mask[x - 1][y]
                    if y - 1 >= 0:
                        surrounding_sum += thumb_mask[x - 1][y - 1]
                    if y + 1 < thumb_mask.shape[1]:
                        surrounding_sum += thumb_mask[x - 1][y + 1]

                if x + 1 < thumb_mask.shape[0]:
                    surrounding_sum += thumb_mask[x + 1][y]
                    if y - 1 >= 0:
                        surrounding_sum += thumb_mask[x + 1][y - 1]
                    if y + 1 < thumb_mask.shape[1]:
                        surrounding_sum += thumb_mask[x + 1][y + 1]

                if y - 1 >= 0:
                    surrounding_sum += thumb_mask[x][y - 1]
                if y + 1 < thumb_mask.shape[1]:
                    surrounding_sum += thumb_mask[x][y + 1]

            else:
                thumb_mask[x][y] = 1

            if surrounding_sum > 5:
                thumb_mask[x][y] = 1

    cords = []
    print("Constructing tiles...")
    for x in range(thumb_mask.shape[0]):
        for y in range(thumb_mask.shape[1]):
            if thumb_mask[x][y] == 1:
                cords.append((x, y))

    pool = Pool(processes=njobs)
    detection_partial = partial(call_region, slide_path, zoom, tile_size)

    tiles = pool.map(detection_partial, cords)

    pool.close()
    pool.join()

    if cords[-1] != (thumb_mask.shape[0] - 1, thumb_mask.shape[1] - 1):
        cords.append(
            (thumb_mask.shape[0] - 1, thumb_mask.shape[1] - 1)
        )  # Append last one if not already there
        print("Appending", cords[-1])  # Append last one if not already there
    return tiles


def call_region(slide_path, zoom, tile_size, cords):
    x = cords[0]
    y = cords[1]
    slide = openslide.open_slide(slide_path)

    region = cv2.cvtColor(
        np.array(
            slide.read_region(
                (y * tile_size[1], x * tile_size[0]), zoom, (tile_size[0], tile_size[1])
            )
        ),
        cv2.COLOR_BGR2RGB,
    )

    cv2.imwrite(
        "/home/grenkoca/Documents/Penn/Summer 2019/outputs/stitched/"
        + str((x, y))
        + ".jpg",
        region,
    )

    # region = downscale(region, (4, 4))
    slide.close()

    return [(x, y), region]


def stitch_tiles(tiles):
    """ Helper method which takes list of tiles and restitches them into whole slide image

    :param tiles: list of tiles
    :return: reconstructed whole slide image
    """

    print("[RECONSTRUCTING MASK]")

    tile_dim = tiles[0][1].shape[0]
    cords = [tile[0] for tile in tiles]

    x_max = max(cord[0] for cord in cords)
    y_max = max(cord[1] for cord in cords)

    image_strips = {}
    print("Creating strips...")
    for x in range(x_max):
        for y in range(y_max):
            if x not in image_strips.keys():
                image_strips[x] = np.array(find_tile(tiles, tuple([x, y]), tile_dim))
            else:
                image_strips[x] = np.concatenate(
                    (image_strips[x], find_tile(tiles, tuple([x, y]), tile_dim)), axis=1
                )

    whole_image = None

    print(
        "Merging strips..."
    )  # TODO: Strip merging slows as it gets bigger (~30%), check runtime?
    for y in image_strips.keys():
        if whole_image is None:
            whole_image = image_strips[y]
        else:
            whole_image = np.concatenate((whole_image, image_strips[y]), axis=0)

    return whole_image


def find_tile(tiles, coords, tile_dims):
    """ Helper method which finds image belonging to coordinates- if none, return blank tile of specified size """

    for tile in tiles:
        if tile[0] == coords:
            return tile[1]
    return np.full((tile_dims, tile_dims, 3), 255)
