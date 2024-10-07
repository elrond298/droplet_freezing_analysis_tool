import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, IntSlider, FloatSlider, Text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
from ipywidgets import IntSlider, FloatSlider, Text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def locate_pcr_tubes(image_path, min_area=100, circularity_threshold=0.2):
    img = cv2.imread(image_path)
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
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def infer_missing_tubes(pcr_tubes, image_shape, tubes_size=(16, 10), rotate='auto'):
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
    print(tubes_size)
    n_rows, n_cols = tubes_size  # Assuming 8x12 grid
    kmeans_x = KMeans(n_clusters=n_cols, random_state=0, n_init=10).fit(rotated_coords[:, 0].reshape(-1, 1))
    kmeans_y = KMeans(n_clusters=n_rows, random_state=0, n_init=10).fit(rotated_coords[:, 1].reshape(-1, 1))

    # Sort cluster centers to get grid lines
    x_lines = np.sort(kmeans_x.cluster_centers_.flatten())
    y_lines = np.sort(kmeans_y.cluster_centers_.flatten())

    # Calculate average radius
    avg_radius = np.mean([tube['radius'] for tube in pcr_tubes])

    # Create a dictionary to store detected tubes
    detected_tubes = {(tube['x'], tube['y']): tube for tube in pcr_tubes}

    # Infer missing tubes
    inferred_tubes = []
    for y in y_lines:
        for x in x_lines:
            # Rotate the point back to the original orientation
            original_x, original_y = rotate_point(center, (x, y), np.radians(angle))
            closest_point = (int(original_x), int(original_y))
            if closest_point in detected_tubes:
                continue
            else:
                # Calculate the minimum distance to all detected tubes
                min_distance = min(np.linalg.norm([tube['x'] - original_x, tube['y'] - original_y]) for tube in pcr_tubes)
                if min_distance > avg_radius:
                    inferred_tubes.append({
                        "x": int(original_x),
                        "y": int(original_y),
                        "radius": int(avg_radius),
                        "inferred": True
                    })

    return inferred_tubes


def detect_inner_circles(image, tubes, roi_size=30, radius=5):
    
    def create_circular_mask(h, w, center, radius):
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

    def find_brightest_point(roi, mask):
        # Apply mask to ROI
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask.astype(np.uint8))
        
        # Apply 5x5 average filter
        kernel = np.ones((radius*2, radius*2), np.float32) / (radius ** 2 * 4)
        avg_roi = cv2.filter2D(masked_roi, -1, kernel)
        
        # Find the coordinates of the maximum average intensity pixel
        y, x = np.unravel_index(np.argmax(avg_roi), avg_roi.shape)
        return x, y
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner_circles = []

    for tube in tubes:
        x, y, r = tube['x'], tube['y'], tube['radius']
        
        # Define ROI
        roi_x = max(0, x - roi_size // 2)
        roi_y = max(0, y - roi_size // 2)
        roi = gray[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        
        # Create circular mask
        mask = create_circular_mask(roi_size, roi_size, (roi_size//2, roi_size//2), r)
        
        # Detect circles in ROI
        circles = cv2.HoughCircles(
            roi, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=roi_size//2,
            param1=50, 
            param2=15, 
            minRadius=5, 
            maxRadius=r-2
        )
        
        # if circles is not None:
        #     circles = np.round(circles[0, :]).astype("int")
        #     # Get the circle with maximum radius (assuming it's the most prominent inner circle)
        #     max_circle = max(circles, key=lambda c: c[2])
        #     cx, cy, cr = max_circle
        #     inner_circles.append({
        #         'x': roi_x + cx,
        #         'y': roi_y + cy,
        #         'radius': cr,
        #         'method': 'hough'
        #     })
        # else:
        #     # Find the brightest point within the circular mask using 5x5 average kernel
        #     cx, cy = find_brightest_point(roi, mask)
        #     inner_circles.append({
        #         'x': roi_x + cx,
        #         'y': roi_y + cy,
        #         'radius': 3,  # Set a small default radius for visualization
        #         'method': 'avg_brightness'
        #     })
        
        cx, cy = find_brightest_point(roi, mask)
        inner_circles.append({
            'x': roi_x + cx,
            'y': roi_y + cy,
            'radius': radius,  # Set a small default radius for visualization
            'method': 'avg_brightness'
        })


    return inner_circles


def display_tubes(image_path, min_area, circularity_threshold, tubes_shape, rotate):
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
