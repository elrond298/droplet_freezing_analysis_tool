import datetime
import os
import traceback

import cv2
from PyQt6.QtWidgets import QFileDialog

from gui_services import dump_inner_circles, render_manual_detection_overlay, render_tube_detection_overlay, run_tube_detection


def reset_tube_detection_view(window):
    if hasattr(window, 'update_timer'):
        window.update_timer.stop()

    window.img = None
    window.pcr_tubes = []
    window.inferred_tubes = []
    window.all_tubes = []
    window.inner_circles = []

    window.run_tube_detection_and_render_plot()
    window.append_log_message(
        "Tube detection reset to use the restored original image.",
        window.LOG_TAB_LOCATE,
        window.LOG_LEVEL_INFO,
    )


def run_tube_detection_and_render_plot(window):
    window.ax.clear()
    try:
        min_area = window.min_area_slider.value()
        circularity_threshold = window.circularity_slider.value() / 100
        rotation = window.rotation_input.text()

        if window.processed_image is not None:
            window.img = window.processed_image
        else:
            window.img = window.load_image_from_path(window.sample_image_path, "Sample image", window.LOG_TAB_LOCATE)
            if window.img is None:
                return

        if window.original_image is None:
            raise ValueError("No source image is loaded. Load an image before running tube detection.")

        window.pcr_tubes, window.inferred_tubes, window.all_tubes, window.inner_circles = run_tube_detection(
            window.img,
            min_area,
            circularity_threshold,
            window.tubes_size,
            rotation,
        )
        img_with_tubes = render_tube_detection_overlay(window.img, window.all_tubes, window.inner_circles)

        window.ax.imshow(cv2.cvtColor(img_with_tubes, cv2.COLOR_BGR2RGB))
        window.ax.set_title(
            f"Detected PCR Tubes: {len(window.pcr_tubes)}, Inferred: {len(window.all_tubes) - len(window.pcr_tubes)}",
            pad=12,
        )
        window.ax.axis('off')

        window.append_log_message(
            f"Detected {len(window.pcr_tubes)} PCR tubes",
            window.LOG_TAB_LOCATE,
            window.LOG_LEVEL_INFO,
        )
        window.append_log_message(
            f"Inferred {len(window.all_tubes) - len(window.pcr_tubes)} additional tubes",
            window.LOG_TAB_LOCATE,
            window.LOG_LEVEL_INFO,
        )

    except Exception as error:
        error_msg = f"Error: {str(error)}\n\n{traceback.format_exc()}"
        window.show_plot_error(window.ax, "An error occurred", error_msg)
        window.append_log_message(error_msg, window.LOG_TAB_LOCATE, window.LOG_LEVEL_ERROR)

    finally:
        window.canvas.draw()


def handle_tube_detection_plot_click(window, event):
    if event.inaxes != window.ax:
        return

    x_pos, y_pos = int(event.xdata), int(event.ydata)

    if event.button == 1:
        if window.inner_circles:
            closest_circle = min(
                window.inner_circles,
                key=lambda circle: ((circle['x'] - x_pos) ** 2 + (circle['y'] - y_pos) ** 2) ** 0.5,
            )
            circle_distance = ((closest_circle['x'] - x_pos) ** 2 + (closest_circle['y'] - y_pos) ** 2) ** 0.5
            if circle_distance < 20:
                window.inner_circles.remove(closest_circle)
                window.append_log_message(
                    f"Removed inner circle at ({closest_circle['x']}, {closest_circle['y']})",
                    window.LOG_TAB_LOCATE,
                    window.LOG_LEVEL_INFO,
                )
                redraw_manual_tube_detection_plot(window)

    elif event.button == 3:
        new_circle = {'x': x_pos, 'y': y_pos, 'radius': 10}
        window.inner_circles.append(new_circle)
        window.append_log_message(
            f"Added new inner circle at ({x_pos}, {y_pos})",
            window.LOG_TAB_LOCATE,
            window.LOG_LEVEL_INFO,
        )
        redraw_manual_tube_detection_plot(window)


def redraw_manual_tube_detection_plot(window):
    window.ax.clear()
    img_with_tubes = render_manual_detection_overlay(window.img, window.all_tubes, window.inner_circles)

    window.ax.imshow(cv2.cvtColor(img_with_tubes, cv2.COLOR_BGR2RGB))
    window.ax.set_title(
        f"PCR Tubes: {len(window.pcr_tubes)}, Inner Circles: {len(window.inner_circles)}",
        pad=12,
    )
    window.ax.axis('off')
    window.canvas.draw()
    window.append_log_message(
        f"Redrawn: PCR Tubes: {len(window.pcr_tubes)}, Inner Circles: {len(window.inner_circles)}",
        window.LOG_TAB_LOCATE,
        window.LOG_LEVEL_DEBUG,
    )


def save_detected_inner_circles(window):
    default_filename = f"inner_circles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    default_filepath = os.path.join(".", default_filename)

    file_path, _ = QFileDialog.getSaveFileName(window, "Save Inner Circles", default_filepath, "Pickle Files (*.pkl)")
    if file_path:
        restored_circles = [
            window.restore_circle_to_original_image(circle)
            for circle in window.normalize_inner_circles(window.inner_circles, default_method="manual")
        ]
        dump_inner_circles(file_path, restored_circles)
        window.append_log_message(
            f"Inner circles saved to {file_path}",
            window.LOG_TAB_LOCATE,
            window.LOG_LEVEL_SUCCESS,
        )