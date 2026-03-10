import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from ipywidgets import interact, interactive, fixed, IntSlider, FloatSlider, Text

def locate_pcr_tubes(image, min_area=100, circularity_threshold=0.2):
    """
    Locates PCR tubes in an image based on contour detection and circularity.

    Args:
        image (str or numpy.ndarray): Path to the image file or a numpy array representing the image.
        min_area (int): Minimum area threshold for detecting contours.
        circularity_threshold (float): Circularity threshold to filter contours.

    Returns:
        list: A list of dictionaries containing the center coordinates ('x', 'y') and radius ('radius') of detected PCR tubes.
        numpy.ndarray: The processed image as a numpy array.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pcr_tubes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = max(cv2.arcLength(contour, True), 0.1)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if area > min_area and circularity > circularity_threshold:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            pcr_tubes.append({"x": center[0], "y": center[1], "radius": radius})
    
    return pcr_tubes, img


def _wrap_half_pi(angle):
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def _pairwise_distances(coords):
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(deltas, axis=2)


def _nearest_neighbor_vectors(coords, max_neighbors=8):
    if len(coords) < 2:
        return np.empty((0, 2), dtype=float)

    distances = _pairwise_distances(coords)
    np.fill_diagonal(distances, np.inf)

    nearest_count = min(max_neighbors, len(coords) - 1)
    nearest_indices = np.argpartition(distances, nearest_count, axis=1)[:, :nearest_count]
    deltas = coords[nearest_indices] - coords[:, None, :]
    flattened = deltas.reshape(-1, 2)
    lengths = np.linalg.norm(flattened, axis=1)
    return flattened[lengths > 1e-6]


def _estimate_grid_axes(coords):
    vectors = _nearest_neighbor_vectors(coords)
    if len(vectors) == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), vectors

    lengths = np.linalg.norm(vectors, axis=1)
    distances = _pairwise_distances(coords)
    np.fill_diagonal(distances, np.inf)
    baseline = np.median(np.min(distances, axis=1))
    cutoff = baseline * 1.6 if np.isfinite(baseline) and baseline > 0 else np.percentile(lengths, 35)
    candidate_vectors = vectors[lengths <= cutoff]
    if len(candidate_vectors) == 0:
        candidate_vectors = vectors

    angles = np.mod(np.arctan2(candidate_vectors[:, 1], candidate_vectors[:, 0]), np.pi)
    best_angle = 0.0
    best_error = np.inf

    for angle in angles:
        error = np.mean(
            np.minimum(
                np.abs(_wrap_half_pi(angles - angle)),
                np.abs(_wrap_half_pi(angles - (angle + np.pi / 2))),
            )
        )
        if error < best_error:
            best_error = error
            best_angle = angle

    if abs(np.sin(best_angle)) > abs(np.cos(best_angle)):
        best_angle = np.mod(best_angle + np.pi / 2, np.pi)

    axis_x = np.array([np.cos(best_angle), np.sin(best_angle)])
    axis_y = np.array([-axis_x[1], axis_x[0]])
    return axis_x, axis_y, candidate_vectors


def _estimate_axis_spacing(vectors, axis_primary, axis_secondary):
    if len(vectors) == 0:
        return 1.0

    parallel = np.abs(vectors @ axis_primary)
    perpendicular = np.abs(vectors @ axis_secondary)
    aligned = parallel > 1e-6
    aligned &= perpendicular <= np.maximum(parallel * 0.35, 2.0)

    samples = parallel[aligned]
    if len(samples) == 0:
        samples = parallel[parallel > 1e-6]
    if len(samples) == 0:
        return 1.0
    return float(np.median(samples))


def _estimate_lattice_offset(values, spacing):
    if spacing <= 0 or len(values) == 0:
        return 0.0
    angles = values / spacing * 2 * np.pi
    complex_mean = np.mean(np.exp(1j * angles))
    if np.isclose(np.abs(complex_mean), 0):
        return float(np.min(values))
    phase = np.angle(complex_mean) % (2 * np.pi)
    offset = phase / (2 * np.pi) * spacing
    return float(offset)


def _select_grid_window(raw_indices, grid_size):
    if len(raw_indices) == 0:
        return 0

    candidates = range(int(np.min(raw_indices)) - grid_size, int(np.max(raw_indices)) + grid_size + 1)
    target_center = (grid_size - 1) / 2.0
    best_start = int(np.min(raw_indices))
    best_score = None

    for start in candidates:
        shifted = raw_indices - start
        in_range = (shifted >= 0) & (shifted < grid_size)
        in_count = int(np.sum(in_range))
        if in_count == 0:
            continue

        centered = shifted[in_range]
        center_error = abs(np.mean(centered) - target_center)
        spread_error = abs((np.max(centered) - np.min(centered)) - min(grid_size - 1, np.max(raw_indices) - np.min(raw_indices)))
        score = (-in_count, center_error, spread_error, abs(start))
        if best_score is None or score < best_score:
            best_score = score
            best_start = start

    return best_start


def calculate_rotation_angle(coords):
    """
    Calculates the dominant grid rotation angle from local nearest-neighbor directions.

    Args:
        coords (numpy.ndarray): A 2D array of coordinates (x, y) of the detected PCR tubes.

    Returns:
        float: The calculated rotation angle in degrees, adjusted to be between -45 and 45 degrees.
    """
    axis_x, _, _ = _estimate_grid_axes(np.asarray(coords, dtype=float))
    angle_degrees = np.degrees(np.arctan2(axis_x[1], axis_x[0]))

    if angle_degrees > 45:
        angle_degrees -= 90
    elif angle_degrees < -45:
        angle_degrees += 90

    return angle_degrees


def rotate_point(origin, point, angle):
    """
    Rotates a point counterclockwise by a given angle around a given origin.

    Args:
        origin (tuple): The origin point (ox, oy) around which to rotate.
        point (tuple): The point (px, py) to be rotated.
        angle (float): The angle in radians to rotate the point.

    Returns:
        tuple: The new coordinates (qx, qy) of the rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def infer_missing_tubes(pcr_tubes, image_shape, tubes_size=(16, 10), rotate='auto'):
    """
    Infers the positions of missing PCR tubes based on detected tubes and image shape.

    Args:
        pcr_tubes (list): A list of dictionaries containing the center coordinates ('x', 'y') and radius ('radius') of detected PCR tubes.
        image_shape (tuple): The shape of the image (height, width).
        tubes_size (tuple): The expected size of the PCR tube grid (rows, columns).
        rotate (str or float): The rotation angle in degrees or 'auto' to automatically calculate the angle.

    Returns:
        list: A list of dictionaries containing the inferred positions of missing PCR tubes, including their center coordinates ('x', 'y'), radius ('radius'), and a flag ('inferred') indicating they were inferred.
    """
    if not pcr_tubes:
        return []

    coords = np.array([[tube['x'], tube['y']] for tube in pcr_tubes], dtype=float)
    n_rows, n_cols = tubes_size

    if rotate == 'auto':
        angle = calculate_rotation_angle(coords)
        axis_x, axis_y, neighbor_vectors = _estimate_grid_axes(coords)
    else:
        angle = float(rotate)
        radians = np.radians(angle)
        axis_x = np.array([np.cos(radians), np.sin(radians)])
        axis_y = np.array([-np.sin(radians), np.cos(radians)])
        neighbor_vectors = _nearest_neighbor_vectors(coords)
    print(f"Calculated rotation angle: {angle:.2f} degrees")

    spacing_x = _estimate_axis_spacing(neighbor_vectors, axis_x, axis_y)
    spacing_y = _estimate_axis_spacing(neighbor_vectors, axis_y, axis_x)

    projected_x = coords @ axis_x
    projected_y = coords @ axis_y
    offset_x = _estimate_lattice_offset(projected_x, spacing_x)
    offset_y = _estimate_lattice_offset(projected_y, spacing_y)

    raw_cols = np.rint((projected_x - offset_x) / spacing_x).astype(int)
    raw_rows = np.rint((projected_y - offset_y) / spacing_y).astype(int)

    col_start = _select_grid_window(raw_cols, n_cols)
    row_start = _select_grid_window(raw_rows, n_rows)

    cols = raw_cols - col_start
    rows = raw_rows - row_start
    in_bounds = (cols >= 0) & (cols < n_cols) & (rows >= 0) & (rows < n_rows)
    if not np.any(in_bounds):
        return []

    fit_rows = rows[in_bounds]
    fit_cols = cols[in_bounds]
    fit_coords = coords[in_bounds]
    design = np.column_stack([np.ones(len(fit_coords)), fit_cols, fit_rows])
    params_x, _, _, _ = np.linalg.lstsq(design, fit_coords[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(design, fit_coords[:, 1], rcond=None)

    origin = np.array([params_x[0], params_y[0]])
    step_col = np.array([params_x[1], params_y[1]])
    step_row = np.array([params_x[2], params_y[2]])

    avg_radius = np.mean([tube['radius'] for tube in pcr_tubes])
    occupancy = set()
    for row, col in zip(fit_rows, fit_cols):
        occupancy.add((int(row), int(col)))

    height, width = image_shape[:2]
    inferred_tubes = []
    for row in range(n_rows):
        for col in range(n_cols):
            if (row, col) in occupancy:
                continue

            lattice_point = origin + col * step_col + row * step_row
            original_x, original_y = int(round(lattice_point[0])), int(round(lattice_point[1]))
            if not (0 <= original_x < width and 0 <= original_y < height):
                continue

            min_distance = min(np.linalg.norm([tube['x'] - original_x, tube['y'] - original_y]) for tube in pcr_tubes)
            if min_distance > avg_radius:
                inferred_tubes.append({
                    "x": original_x,
                    "y": original_y,
                    "radius": int(round(avg_radius)),
                    "inferred": True
                })

    return inferred_tubes


def detect_inner_circles(image, tubes, roi_size=30, radius=10):
    """
    Detects PCR buttons as the brightest circular regions within tubes,
    falling back to brightest point if circle detection fails.

    Args:
        image (numpy.ndarray): The input image as a numpy array.
        tubes (list): A list of dictionaries containing the center coordinates ('x', 'y') and radius ('radius') of detected PCR tubes.
        roi_size (int): The size of the Region of Interest (ROI) around each tube.
        radius (int): The expected radius of the inner circle.

    Returns:
        list: A list of dictionaries containing the center coordinates ('x', 'y') and radius ('radius') of the detected inner circles, along with the method used ('brightest_circle' or 'brightest_point').
    """

    def find_brightest_point(roi, mask=None):
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(roi, (5,5), 0)
        if mask is not None:
            blurred = cv2.bitwise_and(blurred, blurred, mask=mask)

        # Find location of maximum intensity
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        return max_loc[0], max_loc[1], 'brightest_point'

    def find_brightest_circle(roi, expected_radius=5):
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(roi, (5,5), 0)

        # Create a mask for searching (exclude edges)
        mask = np.ones_like(roi, dtype=np.uint8)
        border = 3
        mask[0:border, :] = 0
        mask[-border:, :] = 0
        mask[:, 0:border] = 0
        mask[:, -border:] = 0

        # Find local maxima (brightest points)
        max_val = np.max(blurred[mask == 1])
        min_val = np.min(blurred[mask == 1])

        # Threshold to isolate bright regions
        threshold = max_val - (max_val - min_val) * 0.3
        bright_regions = (blurred >= threshold).astype(np.uint8) * mask

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_regions)

        if num_labels < 2:  # If no bright regions found, fall back to brightest point
            return find_brightest_point(roi, mask)

        # Find the best bright region
        best_score = float('-inf')
        best_x, best_y = None, None

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = int(centroids[i][0])
            y = int(centroids[i][1])

            # Skip if area is too small or too large
            if area < np.pi * (expected_radius/2)**2 or area > np.pi * (expected_radius*2)**2:
                continue

            # Calculate average intensity in this region
            region_mask = (labels == i).astype(np.uint8)
            avg_intensity = np.mean(blurred[region_mask == 1])

            # Calculate circularity
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    continue
            else:
                continue

            # Score based on brightness, area similarity to expected, and circularity
            area_score = 1 - abs(area - np.pi * expected_radius**2) / (np.pi * expected_radius**2)
            brightness_score = avg_intensity / 255
            total_score = brightness_score + area_score * 0.3 + circularity * 0.3

            if total_score > best_score:
                best_score = total_score
                best_x, best_y = x, y

        if best_x is None or best_y is None:
            return find_brightest_point(roi, mask)

        return best_x, best_y, 'brightest_circle'

    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    inner_circles = []
    height, width = gray.shape[:2]

    for tube in tubes:
        x, y, r = int(tube['x']), int(tube['y']), int(tube['radius'])

        # Define ROI boundaries
        roi_size_actual = int(roi_size * 1.2)
        roi_x = max(0, x - roi_size_actual//2)
        roi_y = max(0, y - roi_size_actual//2)
        roi_right = min(width, roi_x + roi_size_actual)
        roi_bottom = min(height, roi_y + roi_size_actual)

        # Extract ROI
        roi = gray[roi_y:roi_bottom, roi_x:roi_right]

        # Check if ROI is valid
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            continue

        # Find brightest circle center or fallback to brightest point
        cx, cy, method = find_brightest_circle(roi)

        if cx is not None and cy is not None:
            inner_circles.append({
                'x': roi_x + cx,
                'y': roi_y + cy,
                'radius': radius,
                'method': method
            })

    return inner_circles


def display_tubes(image_path, min_area, circularity_threshold, tubes_shape, rotate):
    """
    Displays the detected and inferred PCR tubes on the image.

    Args:
        image_path (str): Path to the image file.
        min_area (int): Minimum area threshold for detecting contours.
        circularity_threshold (float): Circularity threshold to filter contours.
        tubes_shape (tuple): The expected size of the PCR tube grid (rows, columns).
        rotate (str or float): The rotation angle in degrees or 'auto' to automatically calculate the angle.

    Returns:
        None: Displays the image with detected and inferred PCR tubes using matplotlib.
    """
    min_area, circularity_threshold = min_area, circularity_threshold
    # center_x, center_y = rotation_center
    
    pcr_tubes, img = locate_pcr_tubes(image_path, min_area, circularity_threshold)
    all_tubes = infer_missing_tubes(pcr_tubes, img.shape)
    inner_circles = detect_inner_circles(img, all_tubes)
    
    # for _ in inner_circles:
    #     print(_)
    
    img_with_tubes = img.copy()
    for tube, inner_circle in zip(all_tubes, inner_circles):
        color = (0, 255, 0) if 'inferred' not in tube else (0, 0, 255)  # Green for detected, Red for inferred
        cv2.circle(img_with_tubes, (tube['x'], tube['y']), tube['radius'], color, 2)
        
        if inner_circle:
            cv2.circle(img_with_tubes, (inner_circle['x'], inner_circle['y']), inner_circle['radius'], (0, 0, 0), 1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_tubes, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected PCR Tubes: {len(pcr_tubes)}, Inferred: {len(all_tubes) - len(pcr_tubes)}")
    plt.axis('off')
    plt.show()

# # Assuming your image is named 'pcr_plate.jpg' and is in the same directory as this notebook
# image_path = '200/img/IMG00000000000001975884.png'
# # image_path = 'rotated_image.jpg'
# rotate = 'auto'

# # Create interactive widgets
# interact(
#     display_tubes,
#     image_path=Text(image_path),
#     min_area=IntSlider(min=10, max=500, step=10, value=100, description='Min Area:'),
#     circularity_threshold=FloatSlider(min=0.1, max=1.0, step=0.05, value=0.2, description='Circularity:'),
#     tubes_shape= fixed((8, 12)),
#     rotate = Text(rotate),
# )
