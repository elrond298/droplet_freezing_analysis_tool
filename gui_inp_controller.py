from __future__ import annotations

import datetime
import os
import re
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem

from gui_services import (
    build_inp_curve,
    deserialize_freezing_temperatures,
    extract_valid_freezing_temperatures,
    serialize_freezing_temperatures,
)

if TYPE_CHECKING:
    from gui import InteractivePlot


AUTO_EXPORT_DIRECTORY = os.path.join("exports", "inp_freezing_temperatures")


def add_inp_dataset_from_files(window: InteractivePlot) -> None:
    file_paths, _ = QFileDialog.getOpenFileNames(
        window,
        "Select Freezing Temperatures",
        ".",
        "Text Files (*.txt)",
    )
    if not file_paths:
        window.append_log_message("No freezing-temperature file selected.", window.LOG_TAB_INP, window.LOG_LEVEL_WARNING)
        return

    settings = _parse_inp_settings(window)
    if settings is None:
        return

    custom_label, droplet_volume_ul, dilution_factor = settings
    added_count = 0
    for file_path in file_paths:
        freezing_temperatures, errors = deserialize_freezing_temperatures(file_path)
        for line, error in errors:
            window.append_log_message(
                f"Skipped malformed line in {os.path.basename(file_path)}: {line.strip()} ({error})",
                window.LOG_TAB_INP,
                window.LOG_LEVEL_WARNING,
            )

        temperatures = extract_valid_freezing_temperatures(freezing_temperatures)
        if not temperatures:
            window.append_log_message(
                f"No valid freezing temperatures found in {file_path}",
                window.LOG_TAB_INP,
                window.LOG_LEVEL_WARNING,
            )
            continue

        base_label = custom_label if custom_label and len(file_paths) == 1 else os.path.basename(file_path)
        _append_inp_dataset(
            window,
            label=base_label,
            freezing_values=temperatures,
            droplet_volume_ul=droplet_volume_ul,
            dilution_factor=dilution_factor,
            source=file_path,
        )
        added_count += 1

    if added_count == 0:
        return

    window.inp_dataset_label_input.clear()
    refresh_inp_plot(window)
    window.append_log_message(
        f"Added {added_count} freezing-temperature dataset(s) to the INP plot.",
        window.LOG_TAB_INP,
        window.LOG_LEVEL_SUCCESS,
    )


def add_selected_inp_preset(window: InteractivePlot) -> None:
    preset_path = window.inp_preset_combo.currentData()
    if not isinstance(preset_path, str) or not os.path.isfile(preset_path):
        window.append_log_message("The selected preset file is not available.", window.LOG_TAB_INP, window.LOG_LEVEL_ERROR)
        return

    settings = _parse_inp_settings(window)
    if settings is None:
        return

    custom_label, droplet_volume_ul, dilution_factor = settings
    freezing_temperatures, errors = deserialize_freezing_temperatures(preset_path)
    for line, error in errors:
        window.append_log_message(
            f"Skipped malformed preset line: {line.strip()} ({error})",
            window.LOG_TAB_INP,
            window.LOG_LEVEL_WARNING,
        )

    temperatures = extract_valid_freezing_temperatures(freezing_temperatures)
    if not temperatures:
        window.append_log_message("The selected preset does not contain valid freezing temperatures.", window.LOG_TAB_INP, window.LOG_LEVEL_WARNING)
        return

    _append_inp_dataset(
        window,
        label=custom_label or window.inp_preset_combo.currentText(),
        freezing_values=temperatures,
        droplet_volume_ul=droplet_volume_ul,
        dilution_factor=dilution_factor,
        source=preset_path,
    )
    window.inp_dataset_label_input.clear()
    refresh_inp_plot(window)
    window.append_log_message(
        f"Added preset dataset: {window.inp_preset_combo.currentText()}",
        window.LOG_TAB_INP,
        window.LOG_LEVEL_SUCCESS,
    )


def add_current_analysis_to_inp(
    window: InteractivePlot,
    label: str | None = None,
    droplet_volume_ul: float | None = None,
    dilution_factor: float | None = None,
    auto_export: bool = False,
) -> None:
    temperatures = extract_valid_freezing_temperatures(window.freezing_temperatures)
    if not temperatures:
        window.append_log_message(
            "No reviewed freezing temperatures are available in Analyze Freezing yet.",
            window.LOG_TAB_INP,
            window.LOG_LEVEL_WARNING,
        )
        return

    if droplet_volume_ul is None or dilution_factor is None:
        settings = _parse_inp_settings(window)
        if settings is None:
            return
        custom_label, droplet_volume_ul, dilution_factor = settings
    else:
        custom_label = "" if label is None else label.strip()

    source_label = "Analyze Freezing tab"
    if auto_export:
        export_path = _auto_export_current_analysis_freezing_temperatures(window, custom_label)
        if export_path is None:
            return
        source_label = export_path

    _append_inp_dataset(
        window,
        label=custom_label or source_label,
        freezing_values=temperatures,
        droplet_volume_ul=droplet_volume_ul,
        dilution_factor=dilution_factor,
        source=source_label,
    )
    window.inp_dataset_label_input.clear()
    refresh_inp_plot(window)
    window.tab_widget.setCurrentIndex(3)
    window.append_log_message(
        f"Added {len(temperatures)} reviewed freezing temperatures from Analyze Freezing.",
        window.LOG_TAB_INP,
        window.LOG_LEVEL_SUCCESS,
    )


def remove_selected_inp_dataset(window: InteractivePlot) -> None:
    selected_row = window.inp_dataset_list.currentRow()
    if selected_row < 0 or selected_row >= len(window.inp_datasets):
        window.append_log_message("Select an INP dataset to remove.", window.LOG_TAB_INP, window.LOG_LEVEL_WARNING)
        return

    removed_dataset = window.inp_datasets.pop(selected_row)
    refresh_inp_plot(window)
    window.append_log_message(
        f"Removed INP dataset: {removed_dataset['label']}",
        window.LOG_TAB_INP,
        window.LOG_LEVEL_INFO,
    )


def clear_inp_datasets(window: InteractivePlot) -> None:
    if not window.inp_datasets:
        window.append_log_message("There are no INP datasets to clear.", window.LOG_TAB_INP, window.LOG_LEVEL_WARNING)
        return

    dataset_count = len(window.inp_datasets)
    window.inp_datasets.clear()
    refresh_inp_plot(window)
    window.append_log_message(
        f"Cleared {dataset_count} INP dataset(s).",
        window.LOG_TAB_INP,
        window.LOG_LEVEL_INFO,
    )


def refresh_inp_plot(window: InteractivePlot) -> None:
    _refresh_dataset_list(window)

    if not window.inp_datasets:
        window.show_inp_plot_instructions()
        return

    window.ax_inp.clear()
    all_temperatures: list[float] = []
    for dataset in window.inp_datasets:
        temperatures = np.asarray(dataset['curve_temperatures'], dtype=float)
        concentrations = np.asarray(dataset['inp_concentrations'], dtype=float)
        all_temperatures.extend(temperatures.tolist())

        window.ax_inp.step(
            temperatures,
            concentrations,
            where='post',
            linewidth=2,
            label=f"{dataset['label']} ({dataset['tube_count']} tubes)",
        )
        window.ax_inp.plot(temperatures, concentrations, 'o', markersize=4)

    window.ax_inp.set_title("Cumulative INP Concentration")
    window.ax_inp.set_xlabel("Freezing temperature (°C)")
    window.ax_inp.set_ylabel("INP concentration (mL^-1)")
    window.ax_inp.set_yscale('log')
    window.ax_inp.grid(True, which='both', alpha=0.25)
    window.ax_inp.legend(loc='best')
    window.apply_figure_font_sizes(window.ax_inp)

    if all_temperatures:
        window.ax_inp.set_xlim(max(all_temperatures), min(all_temperatures))

    window.canvas_inp.draw()


def _append_inp_dataset(
    window: InteractivePlot,
    label: str,
    freezing_values: list[float],
    droplet_volume_ul: float,
    dilution_factor: float,
    source: str,
) -> None:
    curve_temperatures, inp_concentrations = build_inp_curve(
        freezing_values,
        droplet_volume_ul=droplet_volume_ul,
        dilution_factor=dilution_factor,
    )
    unique_label = _make_unique_label(window, label)
    dataset = {
        'label': unique_label,
        'source': source,
        'tube_count': len(freezing_values),
        'droplet_volume_ul': droplet_volume_ul,
        'dilution_factor': dilution_factor,
        'curve_temperatures': curve_temperatures.tolist(),
        'inp_concentrations': inp_concentrations.tolist(),
        'freezing_min': float(min(freezing_values)),
        'freezing_max': float(max(freezing_values)),
    }
    window.inp_datasets.append(dataset)


def _parse_inp_settings(window: InteractivePlot) -> tuple[str, float, float] | None:
    custom_label = window.inp_dataset_label_input.text().strip()

    try:
        droplet_volume_ul = float(window.inp_droplet_volume_input.text().strip())
    except ValueError:
        window.append_log_message("Droplet volume must be a number in uL.", window.LOG_TAB_INP, window.LOG_LEVEL_WARNING)
        return None

    try:
        dilution_factor = float(window.inp_dilution_factor_input.text().strip())
    except ValueError:
        window.append_log_message("Dilution factor must be a number.", window.LOG_TAB_INP, window.LOG_LEVEL_WARNING)
        return None

    if droplet_volume_ul <= 0:
        window.append_log_message("Droplet volume must be greater than zero.", window.LOG_TAB_INP, window.LOG_LEVEL_WARNING)
        return None
    if dilution_factor <= 0:
        window.append_log_message("Dilution factor must be greater than zero.", window.LOG_TAB_INP, window.LOG_LEVEL_WARNING)
        return None

    return custom_label, droplet_volume_ul, dilution_factor


def _make_unique_label(window: InteractivePlot, base_label: str) -> str:
    existing_labels = {dataset['label'] for dataset in window.inp_datasets}
    if base_label not in existing_labels:
        return base_label

    index = 2
    while f"{base_label} ({index})" in existing_labels:
        index += 1
    return f"{base_label} ({index})"


def _auto_export_current_analysis_freezing_temperatures(window: InteractivePlot, label: str) -> str | None:
    export_directory = os.path.join(os.getcwd(), AUTO_EXPORT_DIRECTORY)
    try:
        os.makedirs(export_directory, exist_ok=True)
    except OSError as error:
        window.append_log_message(
            f"Could not create auto-export directory: {export_directory} ({error})",
            window.LOG_TAB_INP,
            window.LOG_LEVEL_ERROR,
        )
        return None

    filename_label = _slugify_label(label) if label else "analyze_freezing"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(export_directory, f"{filename_label}_{timestamp}.txt")

    try:
        with open(export_path, 'w', encoding='utf-8') as file_handle:
            file_handle.writelines(serialize_freezing_temperatures(window.freezing_temperatures))
    except OSError as error:
        window.append_log_message(
            f"Could not auto-export freezing temperatures to {export_path} ({error})",
            window.LOG_TAB_INP,
            window.LOG_LEVEL_ERROR,
        )
        return None

    window.append_log_message(
        f"Auto-exported reviewed freezing temperatures to {export_path}",
        window.LOG_TAB_INP,
        window.LOG_LEVEL_INFO,
    )
    return export_path


def _slugify_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip())
    normalized = normalized.strip("_")
    return normalized or "analyze_freezing"


def _refresh_dataset_list(window: InteractivePlot) -> None:
    current_row = window.inp_dataset_list.currentRow()
    window.inp_dataset_list.blockSignals(True)
    window.inp_dataset_list.clear()
    for dataset in window.inp_datasets:
        item = QListWidgetItem(
            f"{dataset['label']} | {dataset['tube_count']} tubes | {dataset['droplet_volume_ul']:.2f} uL | x{dataset['dilution_factor']:.2f}"
        )
        item.setToolTip(
            f"Source: {dataset['source']}\n"
            f"Freezing range: {dataset['freezing_max']:.2f} to {dataset['freezing_min']:.2f} °C\n"
            f"Max plotted INP: {max(dataset['inp_concentrations']):.3g} mL^-1"
        )
        window.inp_dataset_list.addItem(item)
    if window.inp_datasets:
        window.inp_dataset_list.setCurrentRow(min(max(current_row, 0), len(window.inp_datasets) - 1))
    window.inp_dataset_list.blockSignals(False)