import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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


def calculate_rotation_angle(coords):
    """
    Calculates the rotation angle of the PCR tubes based on Principal Component Analysis (PCA).

    Args:
        coords (numpy.ndarray): A 2D array of coordinates (x, y) of the detected PCR tubes.

    Returns:
        float: The calculated rotation angle in degrees, adjusted to be between -45 and 45 degrees.
    """
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(coords)

    # Get the angle of the first principal component
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    
    # Convert to degrees
    angle_degrees = np.degrees(angle)
    
    # Adjust angle to be between -45 and 45 degrees
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

    # Extract coordinates
    coords = np.array([[tube['x'], tube['y']] for tube in pcr_tubes])

    # Calculate rotation angle
    if rotate == 'auto':
        angle = calculate_rotation_angle(coords)
    else:
        angle = float(rotate)
    print(f"Calculated rotation angle: {angle:.2f} degrees")

    # Rotate coordinates
    center = tuple(np.mean(coords, axis=0))
    rotated_coords = np.array([rotate_point(center, (x, y), np.radians(-angle)) for x, y in coords])

    # Perform KMeans clustering separately for x and y coordinates of rotated points
    n_rows, n_cols = tubes_size
    kmeans_x = KMeans(n_clusters=n_cols, random_state=0, n_init=10).fit(rotated_coords[:, 0].reshape(-1, 1))
    kmeans_y = KMeans(n_clusters=n_rows, random_state=0, n_init=10).fit(rotated_coords[:, 1].reshape(-1, 1))

    # Sort cluster centers to get grid lines
    x_lines = np.sort(kmeans_x.cluster_centers_.flatten())
    y_lines = np.sort(kmeans_y.cluster_centers_.flatten())

    # Calculate average radius
    avg_radius = np.mean([tube['radius'] for tube in pcr_tubes])

    # Create a set to store detected tube coordinates for faster lookup
    detected_tubes = {(int(tube['x']), int(tube['y'])) for tube in pcr_tubes}

    # Infer missing tubes
    inferred_tubes = []
    for y in y_lines:
        for x in x_lines:
            # Rotate the point back to the original orientation
            original_x, original_y = rotate_point(center, (x, y), np.radians(angle))
            original_x, original_y = int(original_x), int(original_y)
            
            if (original_x, original_y) not in detected_tubes:
                # Calculate the minimum distance to all detected tubes
                min_distance = min(np.linalg.norm([tube['x'] - original_x, tube['y'] - original_y]) for tube in pcr_tubes)
                if min_distance > avg_radius:
                    inferred_tubes.append({
                        "x": original_x,
                        "y": original_y,
                        "radius": int(avg_radius),
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
