import datetime
import os
import cv2
import numpy as np

import torch
from torch.cuda import is_available

import supervision as sv

from ultralytics import SAM
from ultralytics import YOLO
from ultralytics import RTDETR


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def mask_image(image, masks):
    """

    :param image:
    :param masks:
    :return:
    """
    masked_image = image.copy()
    combined_masks = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for mask in masks:
        combined_masks = np.logical_or(combined_masks,mask).astype(np.uint8)
    masked_image[combined_masks == 1] = 0

    return masked_image

def calculate_slice_parameters(height, width, slices_x=2, slices_y=2, overlap_percentage=0.3):
    """
    Calculates the slice parameters when using SAHI.

    :param height:
    :param width:
    :param slices_x:
    :param slices_y:
    :param overlap_percentage:
    :return:
    """
    # Calculate base slice dimensions
    slice_width = width // slices_x
    slice_height = height // slices_y

    # Calculate overlap
    overlap_width = int(slice_width * overlap_percentage)
    overlap_height = int(slice_height * overlap_percentage)

    # Adjust slice dimensions to include overlap
    adjusted_slice_width = slice_width + overlap_width
    adjusted_slice_height = slice_height + overlap_height

    # Calculate overlap ratios
    overlap_ratio_w = overlap_width / adjusted_slice_width
    overlap_ratio_h = overlap_height / adjusted_slice_height

    return (adjusted_slice_width, adjusted_slice_height), (overlap_ratio_w, overlap_ratio_h)


def mask_to_polygons(masks, logger):
    """

    :param masks:
    :return:
    """
    # Get the contours for each of the masks
    polygons = []
    for mask in masks:

        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon_length = 0.0025 * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon_length, True)
            # Convert the contour to a numpy array and append to the list
            polygons.append(largest_contour.squeeze())

        except Exception as exc:
            continue

    return polygons


def polygons_to_points(polygons, image):
    """
    Convert polygon points to normalized coordinates relative to the image dimensions.

    :param polygons: List of points
    :param image: numpy array
    :return: List of normalized points for each polygon relative to the provided image dim
    """
    # To hold the normalized polygons
    normalized_polygons = []

    for polygon in polygons:
        # Extract x and y coordinates of the polygon points
        polygon_x = polygon[:, 0]
        polygon_y = polygon[:, 1]

        # Normalize x and y coordinates
        normalized_polygon_x = polygon_x / image.shape[1]
        normalized_polygon_y = polygon_y / image.shape[0]

        # Ensure normalized coordinates are within bounds [0, 1]
        normalized_polygon_x = np.clip(normalized_polygon_x, 0, 1)
        normalized_polygon_y = np.clip(normalized_polygon_y, 0, 1)

        # Create a new list with normalized coordinates for the current polygon
        normalized_points = np.column_stack((normalized_polygon_x, normalized_polygon_y)).tolist()
        normalized_polygons.append(normalized_points)

    return normalized_polygons


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class RockAlgorithm:
    """
    Rock detection algorithm

    Utilizes a configuration file containing the various model/algorithm parameters.
    """

    def __init__(self, config: dict):
        """

        :param config:
        """
        self.sam_model = None
        self.yolo_model = None
        self.slicer = None

        self.config = config
        self.device = 'cuda:0' if is_available() else 'cpu'
        print("NOTE: Using device", self.device)

    def initialize(self):
        """
        Initializes the model

        :return:
        """
        # Get the rock detection model path as specified in the config file
        model_path = self.config["model_path"]

        if not os.path.exists(model_path):
            raise Exception(f"ERROR: Model weights not found in ({model_path})!")

        try:
            if "yolo" in self.config['model_type'].lower():
                # Load the model weights
                self.yolo_model = YOLO(model_path)
            elif "rtdetr" in self.config['model_type'].lower():
                self.yolo_model = RTDETR(model_path)
            else:
                raise Exception(f"ERROR: Model type {self.config['model_type']} not recognized!")

        except Exception as e:
            raise Exception(f"ERROR: Could not load model!\n{e}")

        try:
            # Load the SAM model (sam_b, sam_l, sam_h)
            # This will download the file if it doesn't exist
            self.sam_model = SAM(self.config["sam_model_path"])
        except Exception as e:
            raise Exception(f"ERROR: Could not load SAM model!\n{e}")

        print(f"NOTE: Successfully loaded weights {model_path}")

    def setup_slicer(self, frame):
        """
        Creates the SAHI slicer to be used within slicer callback;
        uses the video dimensions to determine slicer parameters.

        :param frame:
        :return:
        """
        if self.slicer is None:

            slice_wh, overlap_ratio_wh = calculate_slice_parameters(frame.shape[0], frame.shape[1])

            self.slicer = sv.InferenceSlicer(callback=self.slicer_callback,
                                             slice_wh=slice_wh,
                                             iou_threshold=0.25,
                                             overlap_ratio_wh=overlap_ratio_wh,
                                             overlap_filter_strategy=sv.OverlapFilter.NON_MAX_MERGE)

    def slicer_callback(self, image_slice):
        """
        Callback function to perform SAHI using supervision

        :param image_slice:
        :return:
        """

        results = self.yolo_model(image_slice, verbose=False, device=self.device)[0]
        # Convert results to supervision standards
        results = sv.Detections.from_ultralytics(results)
        print(f"...done slicer_callback")
        return results

    def apply_smol(self, frame, detections, logger):
        """
        Performs SAHI on a masked frame, where masked regions are areas
        that have already been detected / segmented by initial inference.

        :param frame:
        :param detections:
        :return:
        """

        logger.info(f"Applying SAHI")

        if self.slicer is None:
            # Setup on the first frame
            self.setup_slicer(frame)

        if detections:
            # Mask out the frame where previous detections were
            masked_frame = mask_image(frame, detections.mask)
            logger.info(f"... Created mask image")
            smol_detections = self.slicer(masked_frame)
            logger.info(f"... Created detections for each slice")
            smol_detections = self.apply_sam(masked_frame, smol_detections, logger)
            logger.info(f"... Applied SAM on sliced detections")

            if smol_detections:
                # If any smol detections, merge
                logger.info(f"... Merging SMOL detections with initial detections")
                detections = sv.Detections.merge([detections, smol_detections])

        logger.info(f"... (apply_smol) done")
        return detections

    def apply_sam(self, frame, detections, logger):
        """

        :param frame:
        :param detections:
        :return:
        """
        logger.info(f"Applying SAM")
        if detections:
            # Pass bboxes to SAM, store masks in detections
            bboxes = detections.xyxy
            logger.info(f"...running SAM")
            masks = self.sam_model(frame, bboxes=bboxes, device=self.device)[0]
            logger.info(f"...ran SAM")
            masks = masks.masks.data.cpu().numpy()
            detections.mask = masks.astype(np.uint8)

        logger.info(f"... (apply_sam) done")
        return detections

    def infer(self, original_frame, logger, image_width, image_height) -> list:
        """
        Performs inference on a single frame; if using smol mode, will use the
        slicer callback function to perform SAHI using supervision (detections will
        be aggregated together).

        Detections will be used with SAM to create instance segmentation masks.

        :param original_frame:
        """
        # Parameters in the config file
        iou = self.config["iou_threshold"]
        conf = self.config["model_confidence_threshold"]

        # Perform detection normally
        logger.info(f"Performing rock detection on original image")
        detections = self.yolo_model(original_frame,
                                     iou=iou,
                                     conf=conf,
                                     device=self.device,
                                     verbose=False)[0]
        logger.info(f"... (infer-1) done")

        # Convert results to supervision standards
        detections = sv.Detections.from_ultralytics(detections)
        # Perform segmentations with bboxes
        logger.info(f"Performing SAM on detections")
        detections = self.apply_sam(original_frame, detections, logger)
        logger.info(f"... (infer-2) done")

        #logger.info(torch.cuda.memory_summary())
        #torch.cuda.empty_cache()
        #logger.info(torch.cuda.memory_summary())

        if self.config["smol"]:
            # Perform detections / segmentations using SAHI
            detections = self.apply_smol(original_frame, detections, logger)
        logger.info(f"... (infer-3) done")

        # Do NMM / NMS with all detections (bboxes)
        detections = detections.with_nmm(iou, class_agnostic=True)
        logger.info(f"... (infer-4) done")
        detections = detections.with_nms(iou, class_agnostic=True)
        logger.info(f"... (infer-5) done")

        # Convert to polygons
        polygons = mask_to_polygons(detections.mask, logger)
        logger.info(f"... (infer-6) done")
        # Convert to points
        polygon_points = polygons_to_points(polygons, original_frame)
        logger.info(f"... (infer-7) done")

        #logger.info(torch.cuda.memory_summary())
        #torch.cuda.empty_cache()
        #logger.info(torch.cuda.memory_summary())

        return polygon_points
