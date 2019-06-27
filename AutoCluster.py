import os
import sys
import time
import argparse
from pprint import pprint

import cv2
import numpy as np
from PIL import Image
from src import ClusterToolkit
from src import Normalization


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "input_folder",
        help="path to folder containing scanned whole slide images",
        type=str,
    )
    parser.add_argument(
        "-o", "--outdir", help="output directory", default="./outputs", type=str
    )
    parser.add_argument(
        "-s",
        "--size",
        help="size (in pixels) of tiles for processing",
        type=int,
        default=512,
    )
    parser.add_argument(
        "-z",
        "--zoom",
        help="level of zoom (0 is highest magnification)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-n", "--njobs", help="number of threads to use", type=int, default=1
    )
    parser.add_argument(
        "-i",
        "--niterations",
        help="number of iterations for finding optimal k",
        type=int,
        default=30,
    )
    parser.add_argument(
        "-k",
        "--kclusters",
        help="if < 1, it will automatically detect optimal K",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "-kmin",
        help="minimum # groups to attempt to classify into",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-kmax",
        help="maximum # groups to attempt to classify into",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--normalize_off", help="turns off stain normalization", action="store_true"
    )
    parser.add_argument(
        "--calibration_image",
        help="location of calibration image for stain normalization",
        type=str,
        default="",
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    assert os.path.isdir(args.input_folder), "No folder found!"

    if not args.input_folder.endswith("/"):
        args.input_folder = args.input_folder + "/"

    supported_formats = [
        ".svs",
        ".tif",
        ".vms",
        ".vmu",
        "ndpi",
        "scn",
        "mrxs",
        ".tiff",
        ".svslide",
        ".bif",
    ]
    files = []

    for file in os.listdir(args.input_folder):
        for format in supported_formats:

            if file.endswith(format):
                files.append(args.input_folder + file)
                break

    if len(files) > 0:
        print("\n==============================================")
        print(f"Queueing {len(files)} files:")
        pprint(files)
        print("==============================================\n")

    assert len(files) >= 1, "No files found in directory!"

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    for path in files:

        slide_name = path.split("/")[-1].split(".")[0]

        print(f"Loading {slide_name} now...")
        start = time.time()
        tissue_tiles = ClusterToolkit.detect_tissue(
            path, tile_size=(args.size, args.size), zoom=args.zoom, njobs=args.njobs
        )

        print(f"{len(tissue_tiles)} nonempty tiles found")
        end = time.time()

        print(f"Took {end-start} seconds")

        tissue_tiles = [ClusterToolkit.downscale_tile(tile) for tile in tissue_tiles]

        if not args.normalize_off:
            print("[BEGINNING NORMALIZATION]")
            norm = Normalization.Normalizer()
            if args.calibration_image == "" or not os.path.exists(
                args.calibration_image
            ):  # If left blank, or path doesn't exist...
                print("Using default stain matrix")
                target_mat = Normalization.rgb2od(
                    np.array([[147, 111, 226], [166, 120, 153]])
                )
                norm.set_target_matrix(target_mat)  # Set target staining to default
            else:  # If valid target image is specified, use as target stain
                print(f"Calibrating using image at {args.calibration_image}")
                calib = cv2.imread(args.calibration_image)
                assert calib is not None, "Invalid calibration image"
                norm.fit(calib)

            tissue_tiles = norm.transform_set(tissue_tiles, njobs=args.njobs)

        ct1 = ClusterToolkit.ClusterToolkit()
        ct1.read_images(tissue_tiles)

        if args.kclusters <= 0:
            ct1.find_optimal_k(
                n_jobs=args.njobs,
                n_iterations=args.niterations,
                range_min=args.kmin,
                range_max=args.kmax + 1,
            )
        else:
            ct1.n_clusters = args.kclusters

        ct1.find_clusters()
        classifier = ct1.supercluster()
        masks = ClusterToolkit.create_masks(tissue_tiles, classifier, njobs=args.njobs)

        masked_WSI = ClusterToolkit.stitch_tiles(masks)
        print(masked_WSI)
        print(masked_WSI.shape)
        Image.fromarray(masked_WSI.astype(np.uint8)).save(
            "/home/grenkoca/Documents/Penn/Summer 2019/outputs/masked.tiff",
            format="tiff",
        )
        print("PIL Masked wsi:")

        if not os.path.exists(args.outdir + "/" + slide_name):
            print("Making path")
            os.mkdir(args.outdir + "/" + slide_name)

        cv2.imwrite(
            args.outdir + "/" + slide_name + "/masked_WSI.jpg",
            masked_WSI
        )

        edge_mask = cv2.Canny(np.uint8(masked_WSI), 200, 300)
        cv2.imwrite(args.outdir + "/" + slide_name + "/edge_mask.png", edge_mask)
        print(f"Outputs written to {args.outdir + '/' + slide_name}")

        # Free up memory for next image
        del ct1
        del tissue_tiles
        del classifier
        del masks
        del masked_WSI

    print("Done!")
