import argparse
import os
from collections import defaultdict, deque

import cv2
import csv   # NEW
import numpy as np
from inference.models.utils import get_roboflow_model

import supervision as sv

# Vehicle type mapping (COCO classes)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Color mapping for different vehicle types (hex format for ColorPalette)
VEHICLE_COLORS_HEX = {
    "car": "#FF0000",          # Red
    "truck": "#00FF00",        # Green
    "bus": "#0000FF",          # Blue
    "motorcycle": "#FFFF00",   # Yellow
    "unknown": "#808080",      # Gray (default)
}

# Create color palette from vehicle colors
VEHICLE_COLOR_PALETTE = sv.ColorPalette.from_hex(
    list(VEHICLE_COLORS_HEX.values())
)

# Map vehicle types to color indices in the palette
VEHICLE_COLOR_INDICES = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "unknown": 4,
}

SOURCE = np.array([[25, 210],
    [270, 220],
    [859, 520],
    [35, 520]])

TARGET_WIDTH = 10
TARGET_HEIGHT = 60

# Traffic light ROI coordinates (x1, y1, x2, y2)
# Default values - user should provide their coordinates
TRAFFIC_LIGHT_ROI = None  # Will be set via command line argument

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def detect_traffic_light(frame, roi_coords):
    """
    Detect traffic light color using HSV color space.
    Focuses on yellow light detection as requested.
    
    Args:
        frame: Input frame (BGR format)
        roi_coords: Tuple of (x1, y1, x2, y2) coordinates for traffic light ROI
    
    Returns:
        tuple: (red_on, yellow_on, green_on, status_text)
    """
    if roi_coords is None:
        return False, False, False, "N/A"
    
    x1, y1, x2, y2 = roi_coords
    # Ensure coordinates are within frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return False, False, False, "INVALID_ROI"
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, False, False, "EMPTY_ROI"
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # HSV ranges for traffic light colors
    # Red (wraps around 0, so need two ranges)
    red_lower1 = np.array([0, 80, 80])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 80, 80])
    red_upper2 = np.array([180, 255, 255])
    
    # Yellow (focus color)
    yellow_lower = np.array([20, 80, 80])
    yellow_upper = np.array([35, 255, 255])
    
    # Green
    green_lower = np.array([40, 80, 80])
    green_upper = np.array([80, 255, 255])
    
    # Create masks
    mask_red = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    
    # Count pixels (threshold: at least 2 pixels to consider light on)
    red_pixels = np.count_nonzero(mask_red)
    yellow_pixels = np.count_nonzero(mask_yellow)
    green_pixels = np.count_nonzero(mask_green)
    
    # Priority: Red > Yellow > Green
    red_on = red_pixels > 2
    yellow_on = False
    green_on = False
    
    if not red_on:
        yellow_on = yellow_pixels > 2
        if not yellow_on:
            green_on = green_pixels > 2
    
    # Determine status text
    if red_on:
        status_text = "RED"
    elif yellow_on:
        status_text = "YELLOW"  # Focus on yellow as requested
    elif green_on:
        status_text = "GREEN"
    else:
        status_text = "OFF"
    
    return red_on, yellow_on, green_on, status_text


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Inference and Supervision"
    )
    parser.add_argument(
        "--model_id",
        default="yolov8x-640",
        help="Roboflow model ID",
        type=str,
    )
    parser.add_argument(
        "--roboflow_api_key",
        default=None,
        help="Roboflow API KEY",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--csv_output_path",                     # NEW
        default="speed_log.csv",               # NEW
        help="Path to the CSV file with speed data",  # NEW
        type=str,                                # NEW
    )
    parser.add_argument(
        "--traffic_light_roi",
        default=None,
        help="Traffic light ROI coordinates as 'x1,y1,x2,y2' (e.g., '668,53,680,64')",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    api_key = args.roboflow_api_key
    api_key = os.environ.get("ROBOFLOW_API_KEY", api_key)
    if api_key is None:
        raise ValueError(
            "Roboflow API key is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY environment variable."
        )
    args.roboflow_api_key = api_key

    # Parse traffic light ROI coordinates if provided
    traffic_light_roi = None
    if args.traffic_light_roi:
        try:
            coords = [int(x.strip()) for x in args.traffic_light_roi.split(',')]
            if len(coords) == 4:
                traffic_light_roi = tuple(coords)
                print(f"Traffic light ROI set to: {traffic_light_roi}")
            else:
                print(f"Warning: Invalid traffic light ROI format. Expected 'x1,y1,x2,y2', got: {args.traffic_light_roi}")
        except ValueError as e:
            print(f"Warning: Could not parse traffic light ROI: {e}")

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = get_roboflow_model(model_id=args.model_id, api_key=args.roboflow_api_key)

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    # Use custom color palette for vehicle types
    box_annotator = sv.BoxAnnotator(
        color=VEHICLE_COLOR_PALETTE,
        thickness=thickness,
        color_lookup=sv.ColorLookup.INDEX
    )
    label_annotator = sv.LabelAnnotator(
        color=VEHICLE_COLOR_PALETTE,
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.INDEX
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # NEW: prepare CSV logging
    csv_rows = []  # we will accumulate and write once at the end
    frame_index = 0  # or start at 1 if you prefer

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            # Detect traffic light status
            red_on, yellow_on, green_on, traffic_light_status = detect_traffic_light(frame, traffic_light_roi)
            
            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            # for tracker_id in detections.tracker_id:
            #     if len(coordinates[tracker_id]) < video_info.fps / 2:
            #         labels.append(f"#{tracker_id}")
            #     else:
            #         coordinate_start = coordinates[tracker_id][-1]
            #         coordinate_end = coordinates[tracker_id][0]
            #         distance = abs(coordinate_start - coordinate_end)
            #         time = len(coordinates[tracker_id]) / video_info.fps
            #         speed = distance / time * 3.6
            #         labels.append(f"#{tracker_id} {int(speed)} km/h")

            # NEW: compute speed, get vehicle type, and also log to CSV
            color_lookup_indices = []  # Store color indices for each detection
            for det_idx, tracker_id in enumerate(detections.tracker_id):
                x_curr, y_curr = points[det_idx]

                # Get vehicle type from class_id
                class_id = detections.class_id[det_idx] if hasattr(detections, 'class_id') and detections.class_id is not None else None
                vehicle_type = VEHICLE_CLASSES.get(int(class_id), "unknown") if class_id is not None else "unknown"
                color_idx = VEHICLE_COLOR_INDICES.get(vehicle_type, VEHICLE_COLOR_INDICES["unknown"])
                color_lookup_indices.append(color_idx)

                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    # Not enough history to estimate speed
                    labels.append(f"#{tracker_id} {vehicle_type}")
                    speed_kmh = None
                    distance = None
                    time_s = None
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time_s = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time_s * 3.6
                    speed_kmh = float(speed)
                    labels.append(f"#{tracker_id} {vehicle_type} {int(speed)} km/h")

                # log one row per detection on this frame
                csv_rows.append(
                    {
                        "frame_index": frame_index,
                        "tracker_id": int(tracker_id),
                        "vehicle_type": vehicle_type,
                        "class_id": int(class_id) if class_id is not None else "",
                        "x": int(x_curr),
                        "y": int(y_curr),
                        "distance": distance if distance is not None else "",
                        "time_s": time_s if time_s is not None else "",
                        "speed_kmh": speed_kmh if speed_kmh is not None else "",
                        "traffic_light_status": traffic_light_status,  # NEW: traffic light status
                        "yellow_light": yellow_on,  # NEW: yellow light flag (focus)
                    }
                )

            annotated_frame = frame.copy()
            
            # Draw traffic light ROI and status on frame
            if traffic_light_roi is not None:
                x1, y1, x2, y2 = traffic_light_roi
                # Determine color based on status
                if red_on:
                    roi_color = (0, 0, 255)  # Red in BGR
                elif yellow_on:
                    roi_color = (0, 255, 255)  # Yellow in BGR (focus)
                elif green_on:
                    roi_color = (0, 255, 0)  # Green in BGR
                else:
                    roi_color = (128, 128, 128)  # Gray
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), roi_color, 2)
                # Display traffic light status text
                status_text_display = f"Traffic Light: {traffic_light_status}"
                if yellow_on:
                    status_text_display += " [YELLOW FOCUS]"
                cv2.putText(
                    annotated_frame, 
                    status_text_display, 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    roi_color, 
                    2
                )
            
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            # Use custom colors for boxes based on vehicle type
            custom_color_lookup = np.array(color_lookup_indices) if color_lookup_indices else None
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections, custom_color_lookup=custom_color_lookup
            )
            # Use custom colors for labels based on vehicle type
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels, custom_color_lookup=custom_color_lookup
            )

            sink.write_frame(annotated_frame)
            # cv2.imshow("frame", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
            frame_index += 1  # Increment frame index
        # cv2.destroyAllWindows()
    
    fieldnames = ["frame_index", "tracker_id", "vehicle_type", "class_id", "x", "y", "distance", "time_s", "speed_kmh", "traffic_light_status", "yellow_light"]
    os.makedirs(os.path.dirname(args.csv_output_path), exist_ok=True) if os.path.dirname(args.csv_output_path) else None

    with open(args.csv_output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)