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
    # "unknown": "#808080",      # Gray (default)
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
    # "unknown": 4,
}

SOURCE = np.array([[25, 210],
    [270, 220],
    [859, 520],
    [35, 520]])

TARGET_WIDTH = 10
TARGET_HEIGHT = 60

# Calibration factor: pixels to meters conversion
# This should be calibrated based on your specific setup
# Example: If TARGET_HEIGHT (60 pixels) represents 60 meters, then PIXELS_TO_METERS = 1.0
# Adjust this value based on your actual setup
PIXELS_TO_METERS = 1.0  # Default: 1 pixel = 1 meter (user should calibrate)

# Traffic light ROI coordinates (x1, y1, x2, y2)
# Default values - user should provide their coordinates
TRAFFIC_LIGHT_ROI = np.array([[183, 100], [224, 100], [223, 120], [182, 120]])  # Will be set via command line argument

# Stop line coordinates (horizontal line: [x1, y1], [x2, y2])
STOP_LINE = np.array([
    [25, 210],
    [270, 220]
])
# Traffic light color palettes (hex format)
TRAFFIC_LIGHT_COLORS = {
    "green": ["51a296", "dffbfd", "26636c", "879c95", "8ec9c5", "2a6c5e", "468f7d", "6ea4a3", "72a98e"],
    "yellow": ["bf985a","f7e0d4","e4d0a0","927353","a8854a","fdf2e2","fdfcf1"],
    "red": ["aa622e", "f4e4c5", "aa8f53", "7f6735", "f7c39d", "b8875b", "d58759"],
}

# Convert hex colors to RGB arrays for each category
def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Pre-compute RGB color palettes
TRAFFIC_LIGHT_RGB = {
    "green": np.array([hex_to_rgb(f"#{c}") for c in TRAFFIC_LIGHT_COLORS["green"]]),
    "yellow": np.array([hex_to_rgb(f"#{c}") for c in TRAFFIC_LIGHT_COLORS["yellow"]]),
    "red": np.array([hex_to_rgb(f"#{c}") for c in TRAFFIC_LIGHT_COLORS["red"]]),
}

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


def find_closest_color(pixel_rgb, color_palette, threshold=50):
    """
    Find if a pixel RGB value is close to any color in the palette.
    Uses Euclidean distance in RGB space.
    
    Args:
        pixel_rgb: RGB tuple or array (R, G, B)
        color_palette: Array of RGB colors to match against
        threshold: Maximum distance to consider a match
    
    Returns:
        bool: True if pixel matches any color in palette
    """
    pixel_rgb = np.array(pixel_rgb)
    distances = np.sqrt(np.sum((color_palette - pixel_rgb) ** 2, axis=1))
    return np.min(distances) <= threshold


def detect_traffic_light(frame, roi_coords, color_threshold=50):
    """
    Detect traffic light color by counting pixels that match specific color palettes.
    Uses spatial information: divides ROI into left (red), middle (yellow), right (green) regions.
    This helps distinguish yellow from red since they're in different positions.
    
    Args:
        frame: Input frame (BGR format)
        roi_coords: Polygon coordinates as numpy array [[x1,y1], [x2,y2], ...] or tuple (x1, y1, x2, y2)
        color_threshold: Maximum RGB distance to consider a color match (default: 50)
    
    Returns:
        tuple: (red_on, yellow_on, green_on, status_text)
    """
    if roi_coords is None:
        return False, False, False, "N/A"
    
    h, w = frame.shape[:2]
    
    # Handle polygon ROI (numpy array) or rectangle ROI (tuple)
    if isinstance(roi_coords, np.ndarray) and len(roi_coords.shape) == 2:
        # Polygon ROI
        polygon = roi_coords.astype(np.int32)
        
        # Create a mask for the polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        
        # Get bounding box for the polygon to extract the region
        x, y, w_roi, h_roi = cv2.boundingRect(polygon)
        if w_roi <= 0 or h_roi <= 0:
            return False, False, False, "INVALID_ROI"
        
        # Extract the region
        roi_region = frame[y:y+h_roi, x:x+w_roi]
        mask_region = mask[y:y+h_roi, x:x+w_roi]
        
        # Only process pixels within the polygon
        if roi_region.size == 0:
            return False, False, False, "EMPTY_ROI"
        
        # Convert BGR to RGB for color matching
        roi_rgb = cv2.cvtColor(roi_region, cv2.COLOR_BGR2RGB)
        
        # Get ROI dimensions for spatial division
        roi_h, roi_w = roi_rgb.shape[:2]
        
    else:
        # Rectangle ROI (backward compatibility)
        if isinstance(roi_coords, tuple) and len(roi_coords) == 4:
            x1, y1, x2, y2 = roi_coords
        else:
            return False, False, False, "INVALID_ROI_FORMAT"
        
        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return False, False, False, "INVALID_ROI"
        
        roi_region = frame[y1:y2, x1:x2]
        if roi_region.size == 0:
            return False, False, False, "EMPTY_ROI"
        
        # Convert BGR to RGB for color matching
        roi_rgb = cv2.cvtColor(roi_region, cv2.COLOR_BGR2RGB)
        mask_region = None  # No mask for rectangle ROI
        roi_h, roi_w = roi_rgb.shape[:2]
    
    # Divide ROI into 3 horizontal regions: left (red), middle (yellow), right (green)
    third_w = roi_w // 3
    
    # Left region (red light position)
    left_region = roi_rgb[:, :third_w]
    left_mask = mask_region[:, :third_w] if mask_region is not None else None
    
    # Middle region (yellow light position)
    middle_region = roi_rgb[:, third_w:2*third_w]
    middle_mask = mask_region[:, third_w:2*third_w] if mask_region is not None else None
    
    # Right region (green light position)
    right_region = roi_rgb[:, 2*third_w:]
    right_mask = mask_region[:, 2*third_w:] if mask_region is not None else None
    
    # Count matches in each region with spatial awareness
    def count_matches_in_region(region, mask, color_palette, color_threshold):
        """Count pixels matching a color palette in a specific region."""
        if mask is not None:
            valid_pixels = region[mask > 0]
        else:
            valid_pixels = region.reshape(-1, 3)
        
        if len(valid_pixels) == 0:
            return 0
        
        count = 0
        for pixel in valid_pixels:
            if find_closest_color(pixel, color_palette, color_threshold):
                count += 1
        return count
    
    # Count matches in each region for each color
    # Left region: primarily check for red
    left_red = count_matches_in_region(left_region, left_mask, TRAFFIC_LIGHT_RGB["red"], color_threshold)
    left_yellow = count_matches_in_region(left_region, left_mask, TRAFFIC_LIGHT_RGB["yellow"], color_threshold)
    
    # Middle region: primarily check for yellow (focus)
    middle_red = count_matches_in_region(middle_region, middle_mask, TRAFFIC_LIGHT_RGB["red"], color_threshold)
    middle_yellow = count_matches_in_region(middle_region, middle_mask, TRAFFIC_LIGHT_RGB["yellow"], color_threshold)
    
    # Right region: primarily check for green
    right_green = count_matches_in_region(right_region, right_mask, TRAFFIC_LIGHT_RGB["green"], color_threshold)
    right_yellow = count_matches_in_region(right_region, right_mask, TRAFFIC_LIGHT_RGB["yellow"], color_threshold)
    
    # Use spatial information: check each region for its expected color
    # Also check for yellow in middle region (focus)
    red_count = left_red  # Red should be in left region
    yellow_count = middle_yellow  # Yellow should be in middle region (focus)
    green_count = right_green  # Green should be in right region
    
    # Additional check: if yellow is detected in middle, prioritize it even if red/yellow counts are close
    # This helps distinguish yellow from red
    min_pixel_threshold = 5  # Minimum pixels to consider a light is on
    
    red_on = red_count >= min_pixel_threshold and red_count > (yellow_count * 0.7)  # Red must be significantly more than yellow
    yellow_on = yellow_count >= min_pixel_threshold and (yellow_count > red_count * 0.8 or middle_yellow > left_red)  # Yellow in middle region
    green_on = green_count >= min_pixel_threshold
    
    # Priority: If yellow is detected in middle region, prioritize it
    if yellow_on and middle_yellow >= min_pixel_threshold:
        # Yellow is in middle, so it's likely yellow
        if middle_yellow > left_red and middle_yellow > right_green:
            red_on = False
            green_on = False
            status_text = "YELLOW"
        elif red_count > green_count and red_count > yellow_count * 1.2:
            yellow_on = False
            green_on = False
            status_text = "RED"
        elif green_count > red_count and green_count > yellow_count * 1.2:
            yellow_on = False
            red_on = False
            status_text = "GREEN"
        else:
            status_text = "YELLOW"  # Default to yellow if detected in middle
    elif red_on:
        yellow_on = False
        green_on = False
        status_text = "RED"
    elif green_on:
        yellow_on = False
        red_on = False
        status_text = "GREEN"
    elif yellow_on:
        red_on = False
        green_on = False
        status_text = "YELLOW"
    else:
        red_on, yellow_on, green_on = False, False, False
        status_text = "OFF"
    
    return red_on, yellow_on, green_on, status_text


def visualize_rois(frame, source_polygon, traffic_light_roi):
    """
    Visualize the ROI polygons on a frame for verification.
    Useful for checking if coordinates are correct.
    
    Args:
        frame: Input frame
        source_polygon: SOURCE polygon for vehicle detection zone
        traffic_light_roi: Traffic light ROI (polygon or rectangle)
    
    Returns:
        Annotated frame with polygons drawn
    """
    annotated_frame = frame.copy()
    
    # Draw SOURCE polygon in red
    annotated_frame = sv.draw_polygon(
        scene=annotated_frame, 
        polygon=source_polygon, 
        color=sv.Color(255, 0, 0),  # Red
        thickness=4
    )
    
    # Draw traffic light ROI in green/yellow
    if traffic_light_roi is not None:
        if isinstance(traffic_light_roi, np.ndarray) and len(traffic_light_roi.shape) == 2:
            # Polygon ROI
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=traffic_light_roi,
                color=sv.Color(0, 255, 0),  # Green
                thickness=4
            )
        elif isinstance(traffic_light_roi, tuple) and len(traffic_light_roi) == 4:
            # Rectangle ROI
            x1, y1, x2, y2 = traffic_light_roi
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    
    return annotated_frame


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
        help="Traffic light ROI coordinates as 'x1,y1,x2,y2' (rectangle) or 'x1,y1,x2,y2,x3,y3,x4,y4' (polygon)",
        type=str,
    )
    parser.add_argument(
        "--visualize_first_frame",
        action="store_true",
        help="Visualize the first frame with ROI polygons drawn to verify coordinates",
    )
    parser.add_argument(
        "--color_threshold",
        default=50,
        type=int,
        help="Maximum RGB distance to consider a color match (default: 50)",
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

    # Parse traffic light ROI coordinates if provided, otherwise use default
    traffic_light_roi = TRAFFIC_LIGHT_ROI  # Use default polygon
    if args.traffic_light_roi:
        try:
            coords = [int(x.strip()) for x in args.traffic_light_roi.split(',')]
            if len(coords) == 4:
                # Rectangle format (x1, y1, x2, y2)
                traffic_light_roi = tuple(coords)
                print(f"Traffic light ROI set to rectangle: {traffic_light_roi}")
            elif len(coords) == 8:
                # Polygon format (x1, y1, x2, y2, x3, y3, x4, y4)
                traffic_light_roi = np.array([[coords[0], coords[1]], 
                                             [coords[2], coords[3]], 
                                             [coords[4], coords[5]], 
                                             [coords[6], coords[7]]])
                print(f"Traffic light ROI set to polygon: {traffic_light_roi}")
            else:
                print(f"Warning: Invalid traffic light ROI format. Expected 'x1,y1,x2,y2' or 'x1,y1,x2,y2,x3,y3,x4,y4', got: {args.traffic_light_roi}")
                print(f"Using default ROI: {TRAFFIC_LIGHT_ROI}")
        except ValueError as e:
            print(f"Warning: Could not parse traffic light ROI: {e}")
            print(f"Using default ROI: {TRAFFIC_LIGHT_ROI}")
    else:
        print(f"Using default traffic light ROI (polygon): {TRAFFIC_LIGHT_ROI}")

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

    # Optional: Visualize first frame to verify coordinates
    if args.visualize_first_frame:
        try:
            first_frame = next(frame_generator)
            first_frame_viz = visualize_rois(first_frame, SOURCE, traffic_light_roi)
            sv.plot_image(first_frame_viz)
            print("First frame visualization displayed. Close the window to continue processing.")
            # Reset generator
            frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)
        except StopIteration:
            print("Warning: Could not read first frame for visualization")

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            # Detect traffic light status using pixel counting with color palettes
            red_on, yellow_on, green_on, traffic_light_status = detect_traffic_light(
                frame, traffic_light_roi, color_threshold=args.color_threshold
            )
            
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

            # Transform stop line to top-view coordinates using the same method as vehicle points
            # Stop line is a horizontal line, transform both endpoints
            stop_line_points = view_transformer.transform_points(STOP_LINE.astype(np.float32))
            # Get the y-coordinate of the stop line in top-view (use average of both points)
            stop_line_y_topview = float(np.mean(stop_line_points[:, 1])) if len(stop_line_points) > 0 else None

            # Calculate traffic density (number of vehicles in detection zone)
            traffic_density = len(detections)

            # Calculate distance to front vehicle for each vehicle
            # Vehicles move from high y to low y in top-view, so front vehicle has lower y
            vehicle_positions = {}  # tracker_id -> (x, y) in top-view
            for det_idx, tracker_id in enumerate(detections.tracker_id):
                vehicle_positions[tracker_id] = points[det_idx]

            labels = []
            # NEW: compute speed, get vehicle type, distance to stop line, distance to front vehicle, and log to CSV
            color_lookup_indices = []  # Store color indices for each detection
            for det_idx, tracker_id in enumerate(detections.tracker_id):
                x_curr, y_curr = points[det_idx]

                # Get vehicle type from class_id
                class_id = detections.class_id[det_idx] if hasattr(detections, 'class_id') and detections.class_id is not None else None
                vehicle_type = VEHICLE_CLASSES.get(int(class_id), "unknown") if class_id is not None else "unknown"
                color_idx = VEHICLE_COLOR_INDICES.get(vehicle_type, 0)  # Default to first color if unknown
                color_lookup_indices.append(color_idx)

                # Calculate distance to stop line using same method as speed (y-coordinate difference in top-view)
                distance_to_stop_line = None
                if stop_line_y_topview is not None:
                    # Distance is the difference in y-coordinates (same units as speed calculation)
                    # Positive means vehicle is before stop line (higher y), negative means past it
                    distance_to_stop_line = float(y_curr - stop_line_y_topview)

                # Calculate distance to front vehicle
                # Front vehicle is the one with lower y value (closer to stop line) in the same lane
                distance_to_front_vehicle = None
                current_y = y_curr
                # Find vehicles in front (lower y values, within reasonable x range to be in same lane)
                front_vehicles = []
                for other_tracker_id, (other_x, other_y) in vehicle_positions.items():
                    if other_tracker_id != tracker_id:
                        # Check if vehicle is in front (lower y) and in similar lane (similar x)
                        y_diff = current_y - other_y  # Positive if other is in front
                        x_diff = abs(x_curr - other_x)
                        # Consider vehicles in front if y_diff > 0 and x_diff is reasonable (same lane)
                        if y_diff > 0 and x_diff < TARGET_WIDTH * 2:  # Within 2 lane widths
                            front_vehicles.append((other_tracker_id, other_y, y_diff))
                
                if front_vehicles:
                    # Find the closest front vehicle (smallest y difference)
                    closest_front = min(front_vehicles, key=lambda v: v[2])
                    distance_to_front_vehicle = float(closest_front[2])  # y-coordinate difference

                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    # Not enough history to estimate speed
                    labels.append(f"#{tracker_id} {vehicle_type}")
                    speed_kmh = None
                    speed_ms = None
                    distance = None
                    time_s = None
                    ttc = None
                else:
                    # Calculate speed using same method: y-coordinate difference over time
                    coordinate_start = coordinates[tracker_id][-1]  # Most recent (highest y)
                    coordinate_end = coordinates[tracker_id][0]    # Oldest (lowest y)
                    distance = abs(coordinate_start - coordinate_end)  # Distance in top-view pixels
                    time_s = len(coordinates[tracker_id]) / video_info.fps
                    speed_pixels_per_sec = distance / time_s
                    # Convert to km/h: pixels/sec * (meters/pixel) * (km/m) * (sec/hour)
                    # Assuming 1 pixel = PIXELS_TO_METERS meters
                    speed_ms = speed_pixels_per_sec * PIXELS_TO_METERS  # m/s
                    speed_kmh = speed_ms * 3.6  # km/h
                    labels.append(f"#{tracker_id} {vehicle_type} {int(speed_kmh)} km/h")
                    
                    # Calculate TTC (Time to Collision) = distance to stop line / speed
                    # Only calculate if vehicle is before stop line and has positive speed
                    if distance_to_stop_line is not None and distance_to_stop_line > 0 and speed_ms is not None and speed_ms > 0:
                        # Convert distance to meters (same units as speed_ms)
                        distance_to_stop_line_m = distance_to_stop_line * PIXELS_TO_METERS
                        ttc = distance_to_stop_line_m / speed_ms  # seconds
                    else:
                        ttc = None

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
                        "speed_ms": speed_ms if speed_ms is not None else "",
                        "traffic_light_status": traffic_light_status,
                        "yellow_light": yellow_on,
                        "distance_to_stop_line": distance_to_stop_line if distance_to_stop_line is not None else "",
                        "distance_to_front_vehicle": distance_to_front_vehicle if distance_to_front_vehicle is not None else "",
                        "traffic_density": traffic_density,
                        "ttc": ttc if ttc is not None else "",
                    }
                )

            annotated_frame = frame.copy()
            
            # Draw SOURCE polygon for verification
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, 
                polygon=SOURCE, 
                color=sv.Color(255, 0, 0),  # Red
                thickness=4
            )
            
            # Draw traffic light ROI and status on frame
            if traffic_light_roi is not None:
                # Determine color based on status
                if red_on:
                    roi_color = sv.Color(255, 0, 0)  # Red
                elif yellow_on:
                    roi_color = sv.Color(255, 255, 0)  # Yellow (focus)
                elif green_on:
                    roi_color = sv.Color(0, 255, 0)  # Green
                else:
                    roi_color = sv.Color(128, 128, 128)  # Gray
                
                # Draw polygon ROI if it's a numpy array, otherwise draw rectangle
                if isinstance(traffic_light_roi, np.ndarray) and len(traffic_light_roi.shape) == 2:
                    # Polygon ROI
                    annotated_frame = sv.draw_polygon(
                        scene=annotated_frame,
                        polygon=traffic_light_roi,
                        color=roi_color,
                        thickness=4
                    )
                else:
                    # Rectangle ROI (backward compatibility)
                    if isinstance(traffic_light_roi, tuple) and len(traffic_light_roi) == 4:
                        x1, y1, x2, y2 = traffic_light_roi
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), roi_color.as_bgr(), 2)
                
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
                    roi_color.as_bgr(), 
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
    
    fieldnames = [
        "frame_index", "tracker_id", "vehicle_type", "class_id", "x", "y", 
        "distance", "time_s", "speed_kmh", "speed_ms",
        "traffic_light_status", "yellow_light",
        "distance_to_stop_line", "distance_to_front_vehicle", "traffic_density", "ttc"
    ]
    os.makedirs(os.path.dirname(args.csv_output_path), exist_ok=True) if os.path.dirname(args.csv_output_path) else None

    with open(args.csv_output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)