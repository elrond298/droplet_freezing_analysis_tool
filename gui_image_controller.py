from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from gui_services import crop_rotated_image, rotate_image

if TYPE_CHECKING:
    from gui import InteractivePlot


def load_selected_image_into_preparation_view(window: InteractivePlot) -> None:
    loaded_image = window.load_image_from_path(window.sample_image_path, "Sample image", window.LOG_TAB_PREPARE)
    if loaded_image is None:
        return

    window.ax_crop.clear()
    window.original_image = loaded_image
    window.rotated_image = window.original_image.copy()
    window.processed_image = None
    window.crop_region = None
    window.rotation_params = None
    img_rgb = cv2.cvtColor(window.original_image, cv2.COLOR_BGR2RGB)
    window.ax_crop.imshow(img_rgb)
    window.ax_crop.axis('off')
    window.create_crop_selector()
    window.canvas_crop.draw()


def apply_preparation_image_rotation(window: InteractivePlot) -> None:
    try:
        if window.original_image is None:
            window.append_log_message(
                "No image loaded. Please load an image first.",
                window.LOG_TAB_PREPARE,
                window.LOG_LEVEL_WARNING,
            )
            return

        rotation_angle = float(window.rotation_input_crop.text())
        window.rotated_image, window.rotation_params = rotate_image(window.original_image, rotation_angle)

        window.ax_crop.clear()
        img_rgb = cv2.cvtColor(window.rotated_image, cv2.COLOR_BGR2RGB)
        window.ax_crop.imshow(img_rgb)
        window.ax_crop.axis('off')
        window.create_crop_selector()
        window.canvas_crop.draw()

    except ValueError:
        window.append_log_message(
            "Invalid rotation angle. Please enter a number.",
            window.LOG_TAB_PREPARE,
            window.LOG_LEVEL_WARNING,
        )
    except Exception as error:
        window.append_log_message(f"Error during rotation: {str(error)}", window.LOG_TAB_PREPARE, window.LOG_LEVEL_ERROR)


def restore_original_preparation_image(window: InteractivePlot) -> None:
    if window.original_image is not None:
        window.rotated_image = window.original_image.copy()
        window.processed_image = None
        window.rotation_params = None
        window.ax_crop.clear()
        img_rgb = cv2.cvtColor(window.original_image, cv2.COLOR_BGR2RGB)
        window.ax_crop.imshow(img_rgb)
        window.ax_crop.axis('off')
        window.create_crop_selector()
        window.canvas_crop.draw()
        window.crop_region = None
        window.append_log_message(
            "Restored the original image and cleared crop/rotation state.",
            window.LOG_TAB_PREPARE,
            window.LOG_LEVEL_INFO,
        )
        window.reset_tube_detection_view()


def apply_selected_crop_to_tube_detection(window: InteractivePlot) -> None:
    try:
        if window.rotated_image is None:
            window.append_log_message(
                "No image loaded. Please load an image first.",
                window.LOG_TAB_PREPARE,
                window.LOG_LEVEL_WARNING,
            )
            return

        if window.crop_region is None:
            window.append_log_message(
                "No crop region selected. Please select a region.",
                window.LOG_TAB_PREPARE,
                window.LOG_LEVEL_WARNING,
            )
            return

        cropped_img = crop_rotated_image(window.rotated_image, window.crop_region)
        window.processed_image = cropped_img

        window.append_log_message(f"Crop region: {window.crop_region}", window.LOG_TAB_PREPARE, window.LOG_LEVEL_INFO)
        window.append_log_message(
            f"Cropped image size: {cropped_img.shape[:2]}",
            window.LOG_TAB_PREPARE,
            window.LOG_LEVEL_INFO,
        )

        if window.rotation_params is not None:
            window.append_log_message(
                f"Rotation angle: {window.rotation_params['angle']} degrees",
                window.LOG_TAB_PREPARE,
                window.LOG_LEVEL_INFO,
            )

        if window.auto_open_tube_detection_after_crop:
            window.tab_widget.setCurrentWidget(window.tab1)
        window.run_tube_detection_and_render_plot()
        window.append_log_message(
            "Crop and rotation applied to tube detection",
            window.LOG_TAB_PREPARE,
            window.LOG_LEVEL_SUCCESS,
        )

    except Exception as error:
        window.append_log_message(f"Error during crop: {str(error)}", window.LOG_TAB_PREPARE, window.LOG_LEVEL_ERROR)