from __future__ import annotations

import datetime
import html
from typing import TYPE_CHECKING

from PyQt6.QtGui import QTextCursor

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QTextEdit

    from gui import InteractivePlot


def get_log_widgets(window: InteractivePlot, tab_number: int) -> list[QTextEdit]:
    if tab_number == 0:
        return [
            widget
            for widget in (
                getattr(window, 'log_text_edit_prep', None),
                getattr(window, 'log_text_edit', None),
                getattr(window, 'log_text_edit2', None),
                getattr(window, 'log_text_edit_inp', None),
            )
            if widget is not None
        ]

    widget = get_log_widget(window, tab_number)
    return [widget] if widget is not None else []


def get_log_widget(window: InteractivePlot, tab_number: int) -> QTextEdit | None:
    if tab_number == window.LOG_TAB_PREPARE:
        return getattr(window, 'log_text_edit_prep', None)
    if tab_number == window.LOG_TAB_LOCATE:
        return getattr(window, 'log_text_edit', None)
    if tab_number == window.LOG_TAB_ANALYZE:
        return getattr(window, 'log_text_edit2', None)
    if tab_number == window.LOG_TAB_INP:
        return getattr(window, 'log_text_edit_inp', None)
    return None


def format_log_message(window: InteractivePlot, message: object, level: str) -> str:
    level_style = window.LOG_LEVEL_STYLES[level]
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    escaped_message = html.escape(str(message)).replace('\n', '<br>')
    return (
        f"<span style=\"color:#7d8a98;\">[{timestamp}]</span> "
        f"<span style=\"color:{level_style['badge']}; font-weight:600;\">[{level_style['label']}]</span> "
        f"<span style=\"color:{level_style['text']};\">{escaped_message}</span>"
    )


def write_log_entry(widget: QTextEdit, formatted_message: str) -> None:
    widget.moveCursor(QTextCursor.MoveOperation.End)
    widget.insertHtml(formatted_message)
    widget.insertPlainText("\n")
    widget.moveCursor(QTextCursor.MoveOperation.End)


def append_log_message(window: InteractivePlot, message: object, tab_number: int, level: str) -> None:
    widgets = get_log_widgets(window, tab_number)
    if not widgets:
        return

    formatted_message = format_log_message(window, message, level)
    for widget in widgets:
        write_log_entry(widget, formatted_message)