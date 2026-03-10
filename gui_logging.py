import datetime
import html

from PyQt6.QtGui import QTextCursor


def get_log_widgets(window, tab_number):
    if tab_number == 0:
        return [
            widget
            for widget in (
                getattr(window, 'log_text_edit_prep', None),
                getattr(window, 'log_text_edit', None),
                getattr(window, 'log_text_edit2', None),
            )
            if widget is not None
        ]

    widget = get_log_widget(window, tab_number)
    return [widget] if widget is not None else []


def get_log_widget(window, tab_number):
    if tab_number == window.LOG_TAB_PREPARE:
        return getattr(window, 'log_text_edit_prep', None)
    if tab_number == window.LOG_TAB_LOCATE:
        return getattr(window, 'log_text_edit', None)
    if tab_number == window.LOG_TAB_ANALYZE:
        return getattr(window, 'log_text_edit2', None)
    return None


def format_log_message(window, message, level):
    level_style = window.LOG_LEVEL_STYLES[level]
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    escaped_message = html.escape(str(message)).replace('\n', '<br>')
    return (
        f"<span style=\"color:#7d8a98;\">[{timestamp}]</span> "
        f"<span style=\"color:{level_style['badge']}; font-weight:600;\">[{level_style['label']}]</span> "
        f"<span style=\"color:{level_style['text']};\">{escaped_message}</span>"
    )


def write_log_entry(widget, formatted_message):
    widget.moveCursor(QTextCursor.MoveOperation.End)
    widget.insertHtml(formatted_message)
    widget.insertPlainText("\n")
    widget.moveCursor(QTextCursor.MoveOperation.End)


def append_log_message(window, message, tab_number, level):
    widgets = get_log_widgets(window, tab_number)
    if not widgets:
        return

    formatted_message = format_log_message(window, message, level)
    for widget in widgets:
        write_log_entry(widget, formatted_message)