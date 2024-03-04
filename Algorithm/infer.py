import os
import shutil
import traceback
import concurrent.futures

import cv2
import numpy as np

import tator
import supervision as sv
from ultralytics import YOLO


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def download_frame(api, media, frame_idx, downloaded_frames_dir):
    """

    :param frame_idx:
    :param media:
    :param downloaded_frames_dir:
    :return:
    """
    try:
        # Get the frame from TATOR
        temp = api.get_frame(id=media.id, tile=f"{media.width}x{media.height}", frames=[frame_idx])
        # Move to download directory
        frame_path = f"{downloaded_frames_dir}/{str(frame_idx)}.png"
        shutil.move(temp, frame_path)

        return frame_idx, frame_path

    except Exception as e:
        raise Exception(f"ERROR: Could not get frame {frame_idx} from {media.id}.\n{e}")


def filter_detections(image, annotations, area_thresh=0.005, conf_thresh=0.0):
    """

    :param image:
    :param annotations:
    :param area_thresh:
    :param conf_thresh:
    :return annotations:
    """

    height, width, channels = image.shape
    image_area = height * width

    # Filter by relative area first
    annotations = annotations[(annotations.box_area / image_area) >= area_thresh]

    # Get the box areas
    boxes = annotations.xyxy
    num_boxes = len(boxes)

    is_large_box = np.zeros(num_boxes, dtype=bool)

    for i, box1 in enumerate(boxes):
        x1, y1, x2, y2 = box1

        # Count the number of smaller boxes contained within box1
        contained_count = np.sum(
            (boxes[:, 0] >= x1) &
            (boxes[:, 1] >= y1) &
            (boxes[:, 2] <= x2) &
            (boxes[:, 3] <= y2)
        )

        # Check if box1 is a large box containing at least two smaller boxes
        is_large_box[i] = contained_count >= 3

    annotations = annotations[~is_large_box]

    # Filter by confidence
    annotations = annotations[annotations.confidence > conf_thresh]

    return annotations


def mask_to_polygons(masks):
    """

    :param masks:
    :return:
    """
    # Ensure at least two masks are provided
    if len(masks) < 2:
        raise ValueError("At least two masks are required.")

    # Initialize result_mask with the first mask
    result_mask = masks[0]

    # Keep track of the non-overlapping regions
    non_overlap_regions = result_mask.copy()

    # Iterate over the remaining masks and handle overlap before adding them to result_mask
    for mask in masks[1:]:

        # First erode it
        mask = cv2.erode(mask, None, iterations=6)
        mask = cv2.dilate(mask, None, iterations=1)

        # Find the intersection of the two masks
        intersection = cv2.bitwise_and(result_mask, mask)

        # Remove the intersection from both masks
        mask1_no_overlap = cv2.subtract(result_mask, intersection)
        mask2_no_overlap = cv2.subtract(mask, intersection)

        # Add the non-overlapped masks
        result_mask_no_overlap = cv2.bitwise_or(mask1_no_overlap, mask2_no_overlap)
        non_overlap_regions = cv2.bitwise_or(non_overlap_regions, mask2_no_overlap)
        result_mask = cv2.bitwise_or(result_mask_no_overlap, mask)

    # Find contours in the non-overlapping regions
    contours, _ = cv2.findContours(non_overlap_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours in the mask
    contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the filled contours on the result_mask
    cv2.drawContours(result_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Initialize a list to store polygon points
    polygons = []

    # Iterate over the contours
    for contour in contours:
        # Remove invalid polygons
        if len(contour) <= 2:
            continue
        # Convert the contour to a numpy array and append to the list
        polygons.append(np.squeeze(contour))

    return polygons


def polygons_to_points(polygons, image):
    """

    :param polygons:
    :param image:
    :return:
    """
    normalized_polygons = []

    for polygon in polygons:
        # Extract x and y coordinates of the polygon points
        polygon_x = polygon[:, 0]
        polygon_y = polygon[:, 1]

        # Normalize x and y coordinates
        normalized_polygon_x = polygon_x / image.shape[1]
        normalized_polygon_y = polygon_y / image.shape[0]

        # Create a new list with normalized coordinates for the current polygon
        normalized_points = np.column_stack((normalized_polygon_x, normalized_polygon_y)).tolist()
        normalized_polygons.append(normalized_points)

    return normalized_polygons


def algorithm(token, project_id, media_id, start_at, end_at, conf, iou, debug):
    """

    :param token:
    :param project_id:
    :param media_id:
    :param start_at:
    :param end_at:
    :param conf:
    :param iou:
    :param debug:
    :return:
    """
    # ------------------------------------------------
    # TATOR setup
    # ------------------------------------------------

    try:
        # Setting the api given the token, authentication
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication successful for {api.whoami().username}")

    except Exception as e:
        raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    try:
        # Find the localization type for this project.
        loc_types = api.get_localization_type_list(project_id)
        found_loc = False

        for loc_type in loc_types:
            if loc_type.id == 460 and loc_type.dtype == 'poly':
                found_loc = True
                break

        if not found_loc:
            raise Exception("ERROR: Localization '460' not found.")

    except Exception as e:
        raise Exception(f"ERROR: Could not get correct localization type.\n{e}")

    try:
        # Find the version type for this project.
        version_types = api.get_version_list(project_id)
        found_version = False

        for version_type in version_types:
            if version_type.id == 545:
                found_version = True
                break

        if not found_version:
            raise Exception("ERROR: Version '545' not found.")

    except Exception as e:
        raise Exception(f"ERROR: Could not get correct version type.\n{e}")

    # ------------------------------------------------
    # Directory setup
    # ------------------------------------------------

    # Get the root directory
    root = os.path.dirname(os.path.realpath(__file__))
    root = root.replace("\\", "/")

    # Get the root data directory
    data_dir = f"{root}/Data/"

    # Raw frame location (from TATOR)
    downloaded_frames_dir = f"{data_dir}/Downloaded_Frames"
    os.makedirs(downloaded_frames_dir, exist_ok=True)

    # Rendered frames with detections
    rendered_frames_dir = f"{data_dir}/Rendered_Frames"
    os.makedirs(rendered_frames_dir, exist_ok=True)

    # Model location
    model_path = f"best.pt"

    if not os.path.exists(model_path):
        raise Exception(f"ERROR: Model weights not found in {root}!")

    try:
        # Load it up
        model = YOLO(model_path)
        print(f"NOTE: Successfully loaded weights {model_path}")

        # Create the annotators for segmentation
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoundingBoxAnnotator()

    except Exception as e:
        raise Exception(f"ERROR: Could not load model weights {model_path}.\n{e}")

    # ------------------------------------------------
    # Download media
    # ------------------------------------------------

    try:
        # Get the media, attributes
        media = api.get_media(media_id)
        width = media.width
        height = media.height

        if not 0 < start_at < media.num_frames or start_at > end_at:
            raise Exception(f"ERROR: Start at frame is invalid")

        if not 0 < end_at < media.num_frames or end_at < start_at:
            raise Exception(f"ERROR: End at frame is invalid")

        # Ensure both values are divisible by 32
        height = height // 32 * 32
        width = width // 32 * 32

        # Image size (passed in model)
        imgsz = [height, width]

        # Get a list of frames to make predictions
        frames = np.arange(start_at, end_at + 1).tolist()

        if not frames:
            raise Exception("ERROR: No frames were extracted!")

        print(f"NOTE: Downloading {len(frames)} frames, "
              f"starting at {start_at} through {end_at} for {media.name}...")

        # Setup multithreading for downloading / saving frames
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
            future_to_frame = {executor.submit(download_frame,
                                               api,
                                               media,
                                               frame_idx,
                                               downloaded_frames_dir): frame_idx for frame_idx in frames}

            # Get the results, store  for later
            for future in concurrent.futures.as_completed(future_to_frame):
                frame_idx = future_to_frame[future]
                try:
                    frame_path = future.result()
                    print(f"NOTE: Frame {frame_path[0]} downloaded to {frame_path[1]}")
                except Exception as e:
                    raise Exception(f"ERROR: Could not download frame {frame_idx}.\n{e}")

    except Exception as e:
        raise Exception(f"ERROR: Could not complete media download for {str(media_id)}.\n{e}")

    # -------------------------------------------
    # Make Inferences
    # -------------------------------------------

    try:
        # To store normalized predictions
        localizations = []

        print("NOTE: Performing inference")

        # Generator for model performing inference
        results = model(f"{downloaded_frames_dir}",
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        half=True,
                        augment=False,
                        max_det=2000,
                        verbose=False,
                        retina_masks=True,
                        stream=True)  # generator of Results objects

        # Loop through the results
        with sv.ImageSink(target_dir_path=rendered_frames_dir, overwrite=True) as sink:
            for result in results:

                # Version issue
                result.obb = None

                # Original frame
                frame_name = os.path.basename(result.path)
                frame_id = frame_name.split("_")[-1].split(".")[0]
                original_frame = result.orig_img

                # Convert the results
                detections = sv.Detections.from_ultralytics(result)
                # Filter the detections
                detections = filter_detections(original_frame, detections)

                # Get the masks of the detections
                masks = detections.mask.astype(np.uint8)
                # Convert to polygons
                polygons = mask_to_polygons(masks)
                # Convert to points
                polygon_points = polygons_to_points(polygons, original_frame)

                # Add each of the polygon points to the localization list
                for points in polygon_points:
                    # Specify spec
                    spec = {
                        'type': loc_type.id,
                        'media_id': media.id,
                        'version': version_type.id,
                        'points': points,
                        'frame': int(frame_id),
                        'Label': 'Rock',
                    }
                    localizations.append(spec)

                if debug:
                    # Create rendered results
                    annotated_frame = mask_annotator.annotate(scene=original_frame, detections=detections)
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

                    # Save to render folder
                    sink.save_image(image=annotated_frame, image_name=frame_name)
                    print(f"NOTE: Rendered results for {frame_name}")

        # ------------------------------------------------
        # Upload to TATOR
        # ------------------------------------------------

        print("NOTE: Uploading predictions to TATOR")
        # Total localizations uploaded
        num_uploaded = 0
        # Loop through and upload to TATOR
        for response in tator.util.chunked_create(api.create_localization_list, project_id, body=localizations):
            num_uploaded += 1

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.print_exc())

    if os.path.exists(downloaded_frames_dir):
        shutil.rmtree(downloaded_frames_dir)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    """

    """
    try:

        token = os.getenv("TOKEN")
        project_id = os.getenv("PROJECT_ID")
        media_id = os.getenv("MEDIA_ID")
        start_at = int(os.getenv("START_AT"))
        end_at = int(os.getenv("END_AT"))
        conf = float(os.getenv("CONFIDENCE"))
        iou = float(os.getenv("IOU"))
        debug = os.getenv('DEBUG')

        # Do checks: if empty, exit early
        if None in [token, project_id, media_id, start_at, end_at, conf, iou]:
            raise Exception("ERROR: Missing parameter value(s)!")

        algorithm(token=token,
                  project_id=project_id,
                  media_id=media_id,
                  start_at=int(start_at),
                  end_at=int(end_at),
                  conf=float(conf),
                  iou=float(iou),
                  debug=debug)

        print("Done.")

    except Exception as e:
        print(f"ERROR: Failed to complete inference!\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
