from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal


@dataclass
class ChatMessage:
    sender: Literal["user", "ai"]
    text: str
    attachment_path: str | None = None
    attachment_kind: Literal["image", "text", "camera"] | None = None


@dataclass
class Conversation:
    title: str
    subtitle: str
    messages: list[ChatMessage] = field(default_factory=list)


def build_chat_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        default=None,
        choices=["auto", "high", "medium", "low"],
        help="Kept for compatibility. Chat interface does not use this parameter.",
    )
    parser.add_argument(
        "--camera-index",
        default=0,
        type=int,
        help="Default camera index when opening photo capture dialog.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model selected for display before opening the interface.",
    )
    return parser


def launch_chat_ai_app(*, window_title: str, camera_index: int = 0, app_mode: str = "desktop", selected_model: str | None = None) -> int:
    try:
        from PySide6.QtCore import QByteArray, QSize, QTimer, Qt, Signal
        from PySide6.QtGui import QAction, QCloseEvent, QIcon, QImage, QPainter, QPixmap
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QDialog,
            QFileDialog,
            QFrame,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMenu,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QScrollArea,
            QSizePolicy,
            QSpacerItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("Missing PySide6. Install with: pip install PySide6 opencv-python") from exc

    try:
        import cv2
    except ImportError:
        cv2 = None

    TRANSLATIONS = {
        "en": {
            "new_chat": "New chat",
            "search": "Search chats",
            "history": "Chat history",
            "settings": "Settings",
            "greeting_title": "Hello! 👋",
            "greeting_text": "How can I help today?",
            "input_placeholder": "Type your message...",
            "camera": "Open camera",
            "choose_image": "Choose image",
            "choose_text": "Choose .txt file",
            "general": "General",
            "settings_intro": "Adjust the app appearance and display language.",
            "appearance": "Appearance",
            "appearance_desc": "Choose the display theme for the app.",
            "language": "Language",
            "language_desc": "Choose the display language for the app.",
            "system": "System",
            "dark": "Dark",
            "light": "Light",
            "capture": "Capture",
            "camera_ready": "Camera is ready. Press Capture to add it to the chat.",
            "camera_unavailable": "Unable to open the camera.",
            "camera_missing": "opencv-python is not installed.",
            "txt_title": "File preview",
            "ai_reply_text": "I received the content and will process it in this chat context.",
            "ai_reply_image": "Image received. I will use it as input for this chat.",
            "ai_reply_camera": "Camera snapshot received and added to the chat.",
            "empty_send": "Enter a message before sending.",
            "info_title": "Info",
            "settings_title": "Settings",
            "camera_window": "Camera",
            "attach_image_label": "Selected image",
            "attach_text_label": "Text file",
            "attach_camera_label": "Camera snapshot",
            "mode_badge": "Mode",
            "today": "Today",
            "yesterday": "Yesterday",
            "days_ago": "days ago",
            "english": "English",
            "vietnamese": "Vietnamese",
        },
        "vi": {
            "new_chat": "Chat mới",
            "search": "Tìm kiếm đoạn chat",
            "history": "Lịch sử chat",
            "settings": "Cài đặt",
            "greeting_title": "Chào bạn! 👋",
            "greeting_text": "Hôm nay bạn cần tôi hỗ trợ gì?",
            "input_placeholder": "Nhập tin nhắn...",
            "camera": "Mở camera",
            "choose_image": "Chọn ảnh",
            "choose_text": "Chọn tệp .txt",
            "general": "Chung",
            "settings_intro": "Điều chỉnh giao diện và ngôn ngữ hiển thị của ứng dụng.",
            "appearance": "Giao diện",
            "appearance_desc": "Chọn giao diện hiển thị cho ứng dụng.",
            "language": "Ngôn ngữ",
            "language_desc": "Chọn ngôn ngữ hiển thị cho ứng dụng.",
            "system": "Hệ thống",
            "dark": "Tối",
            "light": "Sang",
            "capture": "Chụp",
            "camera_ready": "Camera đã sẵn sàng. Nhấn Chụp để thêm ảnh vào đoạn chat.",
            "camera_unavailable": "Không mở được camera.",
            "camera_missing": "Chưa cài opencv-python.",
            "txt_title": "Xem trước tệp",
            "ai_reply_text": "Tôi đã nhận nội dung và sẽ xử lý trong ngữ cảnh đoạn chat này.",
            "ai_reply_image": "Đã nhận ảnh. Tôi sẽ dùng ảnh làm dữ liệu đầu vào cho đoạn chat này.",
            "ai_reply_camera": "Đã nhận ảnh chụp từ camera và thêm vào đoạn chat.",
            "empty_send": "Hãy nhập tin nhắn trước khi gửi.",
            "info_title": "Thông tin",
            "settings_title": "Cài đặt",
            "camera_window": "Camera",
            "attach_image_label": "Ảnh đã chọn",
            "attach_text_label": "Tệp văn bản",
            "attach_camera_label": "Ảnh chụp camera",
            "mode_badge": "Chế độ",
            "today": "Hôm nay",
            "yesterday": "Hôm qua",
            "days_ago": "ngày trước",
            "english": "Tiếng Anh",
            "vietnamese": "Tiếng Việt",
        }
    }

    DARK_STYLESHEET = """
    QMainWindow, QDialog, QWidget {
        background: #202123;
        font-family: "Segoe UI", "Arial", sans-serif;
    }
    QWidget#Root,
    QWidget#ChatPanel,
    QWidget#MessagesHost {
        background: #202123;
    }
    QLabel {
        background: transparent;
        color: #ececf1;
    }
    QLabel#Subtle {
        color: #9a9ca8;
    }
    QLabel#Headline {
        font-size: 34px;
        font-weight: 700;
        color: #ffffff;
    }
    QLabel#BrandText {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
    }
    QLabel#SectionTitle {
        color: #ffffff;
        font-size: 15px;
        font-weight: 600;
    }
    QLabel#Attachment {
        color: #d7e0ff;
        font-weight: 600;
    }
    QFrame#Sidebar {
        background: #171717;
        border-right: 1px solid #2b2d31;
    }
    QFrame#SidebarHeader {
        background: transparent;
        border: none;
    }
    QFrame#SidebarButton,
    QFrame#GreetingCard,
    QFrame#SettingsCard {
        background: #1f2023;
        border: 1px solid #30333a;
        border-radius: 20px;
    }
    QFrame#Composer {
        background: #2c2d31;
        border: 1px solid #3a3d45;
        border-radius: 18px;
    }
    QFrame#HistoryPanel {
        background: #1d1f24;
        border: 1px solid #2b2f36;
        border-radius: 20px;
    }
    QFrame#SearchBox {
        background: transparent;
        border: none;
        border-radius: 18px;
    }
    QFrame#SettingsShell {
        background: #17181c;
        border: 1px solid #2d3138;
        border-radius: 28px;
    }
    QFrame#SettingsNav {
        background: #1b1c20;
        border: 1px solid #2f333a;
        border-radius: 22px;
    }
    QFrame#SettingsContent {
        background: #1c1e22;
        border: 1px solid #2f333a;
        border-radius: 22px;
    }
    QFrame#SettingsOptionCard {
        background: #202228;
        border: 1px solid #31353d;
        border-radius: 20px;
    }
    QFrame#HistoryItem {
        background: #23252c;
        border: 1px solid #2f343d;
        border-radius: 14px;
    }
    QFrame#HistoryItem[selected="true"] {
        background: #2c3038;
        border: 1px solid #3a404b;
    }
    QFrame#BubbleUser {
        background: #2b2c31;
        border: 1px solid #343841;
        border-radius: 22px;
    }
    QFrame#BubbleAI {
        background: #26282d;
        border: 1px solid #343841;
        border-radius: 22px;
    }
    QLabel#Avatar {
        min-width: 38px;
        max-width: 38px;
        min-height: 38px;
        max-height: 38px;
        border-radius: 19px;
        border: 1px solid #3a3d45;
        background: #202228;
        color: white;
        font-weight: 700;
        qproperty-alignment: AlignCenter;
    }
    QLineEdit, QPlainTextEdit, QComboBox, QListWidget {
        background: #1a1b1f;
        border: 1px solid #30333a;
        border-radius: 18px;
        color: #ececf1;
        padding: 12px 14px;
    }
    QFrame#Composer QPlainTextEdit {
        background: transparent;
        border: none;
        border-radius: 0;
        color: #ececf1;
        font-size: 15px;
        padding: 2px 4px;
        min-height: 26px;
        max-height: 120px;
    }
    QPlainTextEdit#ComposerInput,
    QPlainTextEdit#ComposerInput QWidget {
        background: transparent;
        border: none;
    }
    QPlainTextEdit {
        padding-top: 14px;
    }
    QPushButton#ModeButton {
        background: #3a3d45;
        border: 1px solid #50545e;
        border-radius: 14px;
        color: #ececf1;
        font-size: 13px;
        font-weight: 600;
        padding: 6px 14px;
        text-align: center;
        min-height: 34px;
    }
    QPushButton#ModeButton:hover {
        background: #454851;
    }
    QListWidget {
        outline: none;
        padding: 0;
        background: transparent;
        border: none;
    }
    QListWidget::item {
        margin: 0;
        padding: 0;
        border: none;
    }
    QPushButton {
        background: #23252b;
        color: #ececf1;
        border: 1px solid #30333a;
        border-radius: 18px;
        padding: 10px 14px;
        text-align: left;
    }
    QPushButton:hover {
        background: #2b2e35;
    }
    QPushButton#SidebarPrimaryButton,
    QPushButton#SidebarFooterButton {
        min-height: 54px;
        padding-left: 16px;
        font-size: 15px;
        font-weight: 600;
        border: none;
    }
    QPushButton#SidebarPrimaryButton {
        background: transparent;
    }
    QPushButton#SidebarFooterButton {
        background: transparent;
    }
    QPushButton#SidebarPrimaryButton:hover,
    QPushButton#SidebarFooterButton:hover,
    QPushButton#SidebarCompactButton:hover,
    QPushButton#SidebarCompactSearchButton:hover,
    QFrame#SearchBox:hover {
        background: #2b2e35;
    }
    QPushButton#SidebarAppButton {
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        min-width: 42px;
        max-width: 42px;
        min-height: 42px;
        max-height: 42px;
    }
    QPushButton#SidebarAppButton:hover {
        background: transparent;
    }
    QPushButton#SidebarCompactButton {
        min-width: 52px;
        max-width: 52px;
        min-height: 52px;
        max-height: 52px;
        padding: 0;
        border-radius: 18px;
        text-align: center;
        border: none;
        background: transparent;
    }
    QPushButton#SidebarCompactSearchButton {
        min-width: 52px;
        max-width: 52px;
        min-height: 52px;
        max-height: 52px;
        padding: 0;
        border-radius: 18px;
        text-align: center;
        border: none;
        background: transparent;
    }
    QPushButton#RoundButton {
        min-width: 38px;
        max-width: 38px;
        min-height: 38px;
        max-height: 38px;
        border-radius: 19px;
        border: 1px solid #3a3d45;
        background: #3a3d45;
        color: #ececf1;
        font-size: 18px;
        padding: 0;
        text-align: center;
    }
    QPushButton#RoundButton:hover {
        background: #50545e;
    }
    QPushButton#SendButton {
        min-width: 38px;
        max-width: 38px;
        min-height: 38px;
        max-height: 38px;
        border-radius: 19px;
        border: none;
        background: #f4f4f5;
        color: #111214;
        font-size: 18px;
        font-weight: 700;
    }
    QPushButton#SendButton:hover {
        background: #ffffff;
    }
    QPushButton#SettingsNavButton {
        background: #2a2d33;
        border: 1px solid #363b44;
        border-radius: 18px;
        text-align: left;
        padding: 14px 18px;
        font-weight: 600;
    }
    QPushButton#SettingsCloseButton {
        background: transparent;
        border: none;
        color: #d7d9df;
        font-size: 18px;
        padding: 0;
    }
    QPushButton#SettingsCloseButton:hover {
        color: #ffffff;
        background: transparent;
    }
    QComboBox#SettingsCombo {
        min-width: 220px;
        min-height: 44px;
        padding: 10px 14px;
        font-size: 15px;
        font-weight: 600;
    }
    QComboBox#SettingsCombo::drop-down {
        border: none;
        width: 0px;
    }
    QComboBox#SettingsCombo::down-arrow {
        image: none;
    }
    QMenu {
        background: #1b1c20;
        border: 1px solid #30333a;
        border-radius: 16px;
        padding: 8px;
    }
    QMenu::item {
        padding: 10px 18px;
        border-radius: 10px;
    }
    QMenu::item:selected {
        background: #2b2e35;
    }
    """

    LIGHT_STYLESHEET = """
    QMainWindow, QDialog, QWidget {
        background: #f3f4f6;
        font-family: "Segoe UI", "Arial", sans-serif;
    }
    QWidget#Root,
    QWidget#ChatPanel,
    QWidget#MessagesHost {
        background: #f3f4f6;
    }
    QLabel {
        background: transparent;
        color: #14161a;
    }
    QLabel#Subtle {
        color: #6b7280;
    }
    QLabel#Headline {
        font-size: 34px;
        font-weight: 700;
        color: #111827;
    }
    QLabel#BrandText {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
    }
    QLabel#SectionTitle {
        color: #111827;
        font-size: 15px;
        font-weight: 600;
    }
    QLabel#Attachment {
        color: #2f55d4;
        font-weight: 600;
    }
    QFrame#Sidebar {
        background: #eceef2;
        border-right: 1px solid #d7dce4;
    }
    QFrame#SidebarHeader {
        background: transparent;
        border: none;
    }
    QFrame#SidebarButton,
    QFrame#GreetingCard,
    QFrame#SettingsCard {
        background: #ffffff;
        border: 1px solid #d7dce4;
        border-radius: 20px;
    }
    QFrame#Composer {
        background: #f0f1f3;
        border: 1px solid #d0d5de;
        border-radius: 18px;
    }
    QFrame#HistoryPanel {
        background: #f8f9fb;
        border: 1px solid #d7dce4;
        border-radius: 20px;
    }
    QFrame#SearchBox {
        background: transparent;
        border: none;
        border-radius: 18px;
    }
    QFrame#SettingsShell {
        background: #eef1f5;
        border: 1px solid #d7dce4;
        border-radius: 28px;
    }
    QFrame#SettingsNav {
        background: #ffffff;
        border: 1px solid #d7dce4;
        border-radius: 22px;
    }
    QFrame#SettingsContent {
        background: #ffffff;
        border: 1px solid #d7dce4;
        border-radius: 22px;
    }
    QFrame#SettingsOptionCard {
        background: #f8fafc;
        border: 1px solid #dfe4eb;
        border-radius: 20px;
    }
    QFrame#HistoryItem {
        background: #ffffff;
        border: 1px solid #d7dce4;
        border-radius: 14px;
    }
    QFrame#HistoryItem[selected="true"] {
        background: #e9edf6;
        border: 1px solid #cad4e2;
    }
    QFrame#BubbleUser {
        background: #ebeff7;
        border: 1px solid #d7dce4;
        border-radius: 22px;
    }
    QFrame#BubbleAI {
        background: #ffffff;
        border: 1px solid #d7dce4;
        border-radius: 22px;
    }
    QLabel#Avatar {
        min-width: 38px;
        max-width: 38px;
        min-height: 38px;
        max-height: 38px;
        border-radius: 19px;
        border: 1px solid #d7dce4;
        background: #ffffff;
        color: #111827;
        font-weight: 700;
        qproperty-alignment: AlignCenter;
    }
    QLineEdit, QPlainTextEdit, QComboBox, QListWidget {
        background: #ffffff;
        border: 1px solid #d7dce4;
        border-radius: 18px;
        color: #111827;
        padding: 12px 14px;
    }
    QFrame#Composer QPlainTextEdit {
        background: transparent;
        border: none;
        border-radius: 0;
        color: #111827;
        font-size: 15px;
        padding: 2px 4px;
        min-height: 26px;
        max-height: 120px;
    }
    QPlainTextEdit#ComposerInput,
    QPlainTextEdit#ComposerInput QWidget {
        background: transparent;
        border: none;
    }
    QPlainTextEdit {
        padding-top: 14px;
    }
    QPushButton#ModeButton {
        background: #e4e7ed;
        border: 1px solid #c8cdd8;
        border-radius: 14px;
        color: #374151;
        font-size: 13px;
        font-weight: 600;
        padding: 6px 14px;
        text-align: center;
        min-height: 34px;
    }
    QPushButton#ModeButton:hover {
        background: #d8dbe4;
    }
    QListWidget {
        outline: none;
        padding: 0;
        background: transparent;
        border: none;
    }
    QListWidget::item {
        margin: 0;
        padding: 0;
        border: none;
    }
    QPushButton {
        background: #ffffff;
        color: #111827;
        border: 1px solid #d7dce4;
        border-radius: 18px;
        padding: 10px 14px;
        text-align: left;
    }
    QPushButton:hover {
        background: #f8fafc;
    }
    QPushButton#SidebarPrimaryButton,
    QPushButton#SidebarFooterButton {
        min-height: 54px;
        padding-left: 16px;
        font-size: 15px;
        font-weight: 600;
        border: none;
    }
    QPushButton#SidebarPrimaryButton {
        background: transparent;
    }
    QPushButton#SidebarFooterButton {
        background: transparent;
    }
    QPushButton#SidebarPrimaryButton:hover,
    QPushButton#SidebarFooterButton:hover,
    QPushButton#SidebarCompactButton:hover,
    QPushButton#SidebarCompactSearchButton:hover,
    QFrame#SearchBox:hover {
        background: #e2e5e9;
    }
    QPushButton#SidebarAppButton {
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        min-width: 42px;
        max-width: 42px;
        min-height: 42px;
        max-height: 42px;
    }
    QPushButton#SidebarAppButton:hover {
        background: transparent;
    }
    QPushButton#SidebarCompactButton {
        min-width: 52px;
        max-width: 52px;
        min-height: 52px;
        max-height: 52px;
        padding: 0;
        border-radius: 18px;
        text-align: center;
        border: none;
        background: transparent;
    }
    QPushButton#SidebarCompactSearchButton {
        min-width: 52px;
        max-width: 52px;
        min-height: 52px;
        max-height: 52px;
        padding: 0;
        border-radius: 18px;
        text-align: center;
        border: none;
        background: transparent;
    }
    QPushButton#RoundButton {
        min-width: 38px;
        max-width: 38px;
        min-height: 38px;
        max-height: 38px;
        border-radius: 19px;
        border: 1px solid #d0d5de;
        background: #e4e7ed;
        color: #374151;
        font-size: 18px;
        padding: 0;
        text-align: center;
    }
    QPushButton#RoundButton:hover {
        background: #d8dbe4;
    }
    QPushButton#SendButton {
        min-width: 38px;
        max-width: 38px;
        min-height: 38px;
        max-height: 38px;
        border-radius: 19px;
        border: none;
        background: #111827;
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
    }
    QPushButton#SendButton:hover {
        background: #0f172a;
    }
    QPushButton#SettingsNavButton {
        background: #eef2f7;
        border: 1px solid #d7dce4;
        border-radius: 18px;
        text-align: left;
        padding: 14px 18px;
        font-weight: 600;
    }
    QPushButton#SettingsCloseButton {
        background: transparent;
        border: none;
        color: #374151;
        font-size: 18px;
        padding: 0;
    }
    QPushButton#SettingsCloseButton:hover {
        color: #111827;
        background: transparent;
    }
    QComboBox#SettingsCombo {
        min-width: 220px;
        min-height: 44px;
        padding: 10px 14px;
        font-size: 15px;
        font-weight: 600;
    }
    QComboBox#SettingsCombo::drop-down {
        border: none;
        width: 0px;
    }
    QComboBox#SettingsCombo::down-arrow {
        image: none;
    }
    QMenu {
        background: #ffffff;
        border: 1px solid #d7dce4;
        border-radius: 16px;
        padding: 8px;
    }
    QMenu::item {
        padding: 10px 18px;
        border-radius: 10px;
    }
    QMenu::item:selected {
        background: #eef2f7;
    }
    """

    def tr(language: str, key: str) -> str:
        return TRANSLATIONS.get(language, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))

    ICONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icons")
    ICON_CACHE: dict[tuple[str, str, int], QIcon] = {}

    def icon(name: str) -> QIcon:
        return QIcon(os.path.join(ICONS_DIR, name))

    def themed_icon(name: str, color: str, size: int) -> QIcon:
        cache_key = (name, color, size)
        if cache_key in ICON_CACHE:
            return ICON_CACHE[cache_key]
        path = Path(ICONS_DIR) / name
        if not path.exists():
            return QIcon()
        svg_text = path.read_text(encoding="utf-8")
        svg_text = svg_text.replace("currentColor", color).replace("#AAB0BC", color)
        renderer = QSvgRenderer(QByteArray(svg_text.encode("utf-8")))
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        result = QIcon(pixmap)
        ICON_CACHE[cache_key] = result
        return result

    def themed_pixmap(name: str, color: str, size: int) -> QPixmap:
        return themed_icon(name, color, size).pixmap(size, size)

    class CameraCaptureDialog(QDialog):
        captured = Signal(str)

        def __init__(self, *, language: str, camera_index_value: int, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.language = language
            self.camera_index_value = camera_index_value
            self.capture = None
            self.latest_frame = None
            self.fps = 0.0
            self.frame_count = 0
            self.last_fps_time = 0.0
            self.setWindowTitle(tr(language, "camera_window"))
            self.setModal(True)
            self.resize(760, 560)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            self.preview_label = QLabel()
            self.preview_label.setMinimumSize(720, 420)
            self.preview_label.setAlignment(Qt.AlignCenter)
            self.preview_label.setObjectName("GreetingCard")
            self.preview_label.setStyleSheet("border-radius: 18px;")
            layout.addWidget(self.preview_label)

            self.status_label = QLabel()
            self.status_label.setObjectName("Subtle")
            layout.addWidget(self.status_label)

            row = QHBoxLayout()
            row.addStretch(1)
            self.capture_button = QPushButton(tr(language, "capture"))
            self.capture_button.clicked.connect(self.capture_frame)
            row.addWidget(self.capture_button)
            layout.addLayout(row)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_preview)
            self.start_camera()

        def start_camera(self) -> None:
            if cv2 is None:
                self.status_label.setText(tr(self.language, "camera_missing"))
                self.capture_button.setEnabled(False)
                return
            self.capture = cv2.VideoCapture(self.camera_index_value, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = None
                self.status_label.setText(tr(self.language, "camera_unavailable"))
                self.capture_button.setEnabled(False)
                return
            self.status_label.setText(tr(self.language, "camera_ready"))
            self.timer.start(30)

        def update_preview(self) -> None:
            import time
            if self.capture is None:
                return
            ok, frame = self.capture.read()
            if not ok:
                self.status_label.setText(tr(self.language, "camera_unavailable"))
                return
            self.latest_frame = frame
            # FPS calculation
            self.frame_count += 1
            now = time.monotonic()
            if self.last_fps_time == 0.0:
                self.last_fps_time = now
            elapsed = now - self.last_fps_time
            if elapsed >= 0.5:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_time = now
            # Draw FPS overlay
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fps_text = f"FPS: {self.fps:.0f}"
            cv2.rectangle(rgb, (8, 8), (110, 36), (0, 0, 0), -1)
            cv2.putText(rgb, fps_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 100), 2, cv2.LINE_AA)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.preview_label.setPixmap(pixmap)

        def capture_frame(self) -> None:
            if self.latest_frame is None or cv2 is None:
                return
            output_dir = Path(tempfile.gettempdir()) / "yolo_chat_captures"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "camera_capture.png"
            cv2.imwrite(str(output_path), self.latest_frame)
            self.captured.emit(str(output_path))
            self.accept()

        def closeEvent(self, event: QCloseEvent) -> None:
            self.timer.stop()
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            super().closeEvent(event)

    class SettingsDialog(QDialog):
        def __init__(self, *, parent_window: "ChatWindow") -> None:
            super().__init__(parent_window)
            self.window = parent_window
            self.setWindowTitle(tr(self.window.language, "settings_title"))
            self.setModal(True)
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.resize(1060, 700)
            self.build_ui()

        def build_ui(self) -> None:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(20, 20, 20, 20)
            outer.setSpacing(0)

            shell = QFrame()
            shell.setObjectName("SettingsShell")
            outer.addWidget(shell)

            shell_layout = QVBoxLayout(shell)
            shell_layout.setContentsMargins(24, 22, 24, 24)
            shell_layout.setSpacing(18)

            header_row = QHBoxLayout()
            header_row.setSpacing(12)
            self.dialog_title = QLabel(tr(self.window.language, "settings_title"))
            self.dialog_title.setStyleSheet("font-size: 28px; font-weight: 700;")
            header_row.addWidget(self.dialog_title)
            header_row.addStretch(1)
            close_button = QPushButton("✕")
            close_button.setObjectName("SettingsCloseButton")
            close_button.setFixedSize(28, 28)
            close_button.clicked.connect(self.accept)
            header_row.addWidget(close_button)
            shell_layout.addLayout(header_row)

            body_row = QHBoxLayout()
            body_row.setSpacing(18)

            sidebar = QFrame()
            sidebar.setObjectName("SettingsNav")
            sidebar_layout = QVBoxLayout(sidebar)
            sidebar_layout.setContentsMargins(18, 18, 18, 18)
            sidebar_layout.setSpacing(14)

            self.general_button = QPushButton()
            self.general_button.setObjectName("SettingsNavButton")
            self.general_button.setMinimumHeight(52)
            sidebar_layout.addWidget(self.general_button)
            sidebar_layout.addStretch(1)
            body_row.addWidget(sidebar, 2)

            content = QFrame()
            content.setObjectName("SettingsContent")
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(28, 26, 28, 26)
            content_layout.setSpacing(18)

            self.section_title = QLabel()
            self.section_title.setObjectName("SectionTitle")
            self.section_title.setStyleSheet("font-size: 30px; font-weight: 700;")
            content_layout.addWidget(self.section_title)

            self.section_intro = QLabel()
            self.section_intro.setObjectName("Subtle")
            self.section_intro.setWordWrap(True)
            self.section_intro.setStyleSheet("font-size: 15px;")
            content_layout.addWidget(self.section_intro)

            self.appearance_card = QFrame()
            self.appearance_card.setObjectName("SettingsOptionCard")
            appearance_card_layout = QHBoxLayout(self.appearance_card)
            appearance_card_layout.setContentsMargins(22, 20, 22, 20)
            appearance_card_layout.setSpacing(18)
            appearance_text = QVBoxLayout()
            appearance_text.setSpacing(6)
            self.appearance_label = QLabel()
            self.appearance_label.setStyleSheet("font-size: 18px; font-weight: 700;")
            self.appearance_desc = QLabel()
            self.appearance_desc.setObjectName("Subtle")
            self.appearance_desc.setWordWrap(True)
            appearance_text.addWidget(self.appearance_label)
            appearance_text.addWidget(self.appearance_desc)
            appearance_card_layout.addLayout(appearance_text, 1)

            self.theme_combo = QComboBox()
            self.theme_combo.setObjectName("SettingsCombo")
            self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)
            appearance_card_layout.addWidget(self.theme_combo, 0, Qt.AlignVCenter)
            content_layout.addWidget(self.appearance_card)

            self.language_card = QFrame()
            self.language_card.setObjectName("SettingsOptionCard")
            language_card_layout = QHBoxLayout(self.language_card)
            language_card_layout.setContentsMargins(22, 20, 22, 20)
            language_card_layout.setSpacing(18)
            language_text = QVBoxLayout()
            language_text.setSpacing(6)
            self.language_label = QLabel()
            self.language_label.setStyleSheet("font-size: 18px; font-weight: 700;")
            self.language_desc = QLabel()
            self.language_desc.setObjectName("Subtle")
            self.language_desc.setWordWrap(True)
            language_text.addWidget(self.language_label)
            language_text.addWidget(self.language_desc)
            language_card_layout.addLayout(language_text, 1)

            self.language_combo = QComboBox()
            self.language_combo.setObjectName("SettingsCombo")
            self.language_combo.currentIndexChanged.connect(self.on_language_changed)
            language_card_layout.addWidget(self.language_combo, 0, Qt.AlignVCenter)
            content_layout.addWidget(self.language_card)

            content_layout.addStretch(1)
            body_row.addWidget(content, 5)
            shell_layout.addLayout(body_row)
            self.retranslate_dialog()

        def on_theme_changed(self, index: int) -> None:
            self.window.theme_mode = {0: "system", 1: "dark", 2: "light"}[index]
            self.window.apply_theme()

        def on_language_changed(self, index: int) -> None:
            self.window.language = "en" if index == 0 else "vi"
            self.window.retranslate_ui()
            self.retranslate_dialog()

        def retranslate_dialog(self) -> None:
            language = self.window.language
            self.setWindowTitle(tr(language, "settings_title"))
            self.dialog_title.setText(tr(language, "settings_title"))
            self.general_button.setText(tr(language, "general"))
            self.section_title.setText(tr(language, "general"))
            self.section_intro.setText(tr(language, "settings_intro"))
            self.appearance_label.setText(tr(language, "appearance"))
            self.appearance_desc.setText(tr(language, "appearance_desc"))
            self.language_label.setText(tr(language, "language"))
            self.language_desc.setText(tr(language, "language_desc"))

            theme_index = {"system": 0, "dark": 1, "light": 2}.get(self.window.theme_mode, 0)
            self.theme_combo.blockSignals(True)
            self.theme_combo.clear()
            self.theme_combo.addItems([tr(language, "system"), tr(language, "dark"), tr(language, "light")])
            self.theme_combo.setCurrentIndex(theme_index)
            self.theme_combo.blockSignals(False)

            language_index = 0 if self.window.language == "en" else 1
            self.language_combo.blockSignals(True)
            self.language_combo.clear()
            self.language_combo.addItems([tr(language, "english"), tr(language, "vietnamese")])
            self.language_combo.setCurrentIndex(language_index)
            self.language_combo.blockSignals(False)

    class HistoryItemWidget(QFrame):
        def __init__(
            self,
            title: str,
            subtitle: str,
            *,
            icon_color: str,
            selected: bool = False,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("HistoryItem")
            self.setProperty("selected", selected)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(10)

            icon_label = QLabel()
            icon_label.setPixmap(themed_pixmap("chat_history.svg", icon_color, 20))
            icon_label.setFixedSize(20, 20)
            layout.addWidget(icon_label, 0, Qt.AlignTop)

            text_layout = QVBoxLayout()
            text_layout.setSpacing(4)

            title_label = QLabel(title)
            title_label.setStyleSheet("font-size: 15px; font-weight: 600;")
            title_label.setWordWrap(True)
            text_layout.addWidget(title_label)

            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("Subtle")
            text_layout.addWidget(subtitle_label)
            layout.addLayout(text_layout, 1)
            self.setFixedHeight(68)

    class ChatBubble(QFrame):
        def __init__(self, message: ChatMessage, *, language: str, align_right: bool, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            outer = QHBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(12)

            if align_right:
                outer.addStretch(1)
            else:
                avatar = QLabel("AI")
                avatar.setObjectName("Avatar")
                outer.addWidget(avatar, alignment=Qt.AlignTop)

            bubble = QFrame()
            bubble.setObjectName("BubbleUser" if align_right else "BubbleAI")
            bubble.setMaximumWidth(760)
            bubble_layout = QVBoxLayout(bubble)
            bubble_layout.setContentsMargins(18, 14, 18, 14)
            bubble_layout.setSpacing(8)

            if message.attachment_path:
                attachment_key = {
                    "image": "attach_image_label",
                    "text": "attach_text_label",
                    "camera": "attach_camera_label",
                }.get(message.attachment_kind, "attach_image_label")
                attachment_label = QLabel(
                    f"{tr(language, attachment_key)}: {Path(message.attachment_path).name}"
                )
                attachment_label.setObjectName("Attachment")
                bubble_layout.addWidget(attachment_label)
                if message.attachment_kind in {"image", "camera"}:
                    pixmap = QPixmap(message.attachment_path)
                    if not pixmap.isNull():
                        preview = QLabel()
                        preview.setPixmap(pixmap.scaled(280, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        bubble_layout.addWidget(preview)

            text_label = QLabel(message.text)
            text_label.setWordWrap(True)
            text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            text_label.setStyleSheet("font-size: 15px;")
            bubble_layout.addWidget(text_label)
            outer.addWidget(bubble, 0, Qt.AlignTop)

            if not align_right:
                outer.addStretch(1)

    class ChatWindow(QMainWindow):
        def __init__(self, *, title: str, initial_camera_index: int, mode_label: str, model_label: str | None) -> None:
            super().__init__()
            self.language = "vi"
            self.theme_mode = "dark"
            self.effective_theme = "dark"
            self.sidebar_expanded = True
            self.is_refreshing_history = False
            self.initial_camera_index = initial_camera_index
            self.mode_label = mode_label
            self.model_label = model_label or ""
            self.conversations: list[Conversation] = []
            self.active_conversation_index = 0
            self.setWindowTitle(title)
            self.resize(1480, 920)
            self.build_ui()
            self.seed_default_conversations()
            self.apply_theme()
            self.retranslate_ui()

        def build_ui(self) -> None:
            root_widget = QWidget()
            root_widget.setObjectName("Root")
            root = QHBoxLayout(root_widget)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
            self.setCentralWidget(root_widget)

            self.sidebar = QFrame()
            self.sidebar.setObjectName("Sidebar")
            self.sidebar.setMinimumWidth(92)
            self.sidebar.setMaximumWidth(360)
            sidebar_layout = QVBoxLayout(self.sidebar)
            sidebar_layout.setContentsMargins(16, 16, 16, 16)
            sidebar_layout.setSpacing(14)

            header_frame = QFrame()
            header_frame.setObjectName("SidebarHeader")
            header = QHBoxLayout(header_frame)
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(12)
            self.brand_text = QLabel("Chat AI")
            self.brand_text.setObjectName("BrandText")
            self.sidebar_app_button = QPushButton()
            self.sidebar_app_button.setObjectName("SidebarAppButton")
            self.sidebar_app_button.setIconSize(QSize(28, 28))
            self.sidebar_app_button.clicked.connect(self.toggle_sidebar)
            header.addWidget(self.brand_text)
            header.addStretch(1)
            header.addWidget(self.sidebar_app_button, 0, Qt.AlignRight)
            sidebar_layout.addWidget(header_frame)

            self.new_chat_button = QPushButton()
            self.new_chat_button.setObjectName("SidebarPrimaryButton")
            self.new_chat_button.setIconSize(QSize(20, 20))
            self.new_chat_button.clicked.connect(self.start_new_chat)
            sidebar_layout.addWidget(self.new_chat_button)

            self.search_box = QFrame()
            self.search_box.setObjectName("SearchBox")
            search_layout = QHBoxLayout(self.search_box)
            search_layout.setContentsMargins(14, 10, 14, 10)
            search_layout.setSpacing(10)
            self.search_icon = QLabel()
            self.search_icon.setFixedSize(18, 18)
            search_layout.addWidget(self.search_icon)
            self.search_input = QLineEdit()
            self.search_input.setFrame(False)
            self.search_input.setStyleSheet("border: none; background: transparent; padding: 0;")
            self.search_input.textChanged.connect(self.refresh_history)
            search_layout.addWidget(self.search_input, 1)
            sidebar_layout.addWidget(self.search_box)

            self.search_compact_button = QPushButton()
            self.search_compact_button.setObjectName("SidebarCompactSearchButton")
            self.search_compact_button.setIconSize(QSize(22, 22))
            self.search_compact_button.setToolTip(tr(self.language, "search"))
            self.search_compact_button.clicked.connect(self.focus_search)
            sidebar_layout.addWidget(self.search_compact_button, 0, Qt.AlignHCenter)

            self.history_title = QLabel()
            self.history_title.setObjectName("SectionTitle")
            sidebar_layout.addWidget(self.history_title)

            self.history_panel = QFrame()
            self.history_panel.setObjectName("HistoryPanel")
            history_panel_layout = QVBoxLayout(self.history_panel)
            history_panel_layout.setContentsMargins(10, 10, 10, 10)
            history_panel_layout.setSpacing(0)

            self.history_list = QListWidget()
            self.history_list.setFrameShape(QFrame.NoFrame)
            self.history_list.setSpacing(10)
            self.history_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
            self.history_list.currentRowChanged.connect(self.select_conversation)
            history_panel_layout.addWidget(self.history_list)
            sidebar_layout.addWidget(self.history_panel)

            self.sidebar_spacer = QWidget()
            self.sidebar_spacer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            sidebar_layout.addWidget(self.sidebar_spacer)

            self.settings_button = QPushButton()
            self.settings_button.setObjectName("SidebarFooterButton")
            self.settings_button.setIconSize(QSize(20, 20))
            self.settings_button.clicked.connect(self.open_settings)
            sidebar_layout.addWidget(self.settings_button)
            root.addWidget(self.sidebar, 1)

            self.chat_panel = QWidget()
            self.chat_panel.setObjectName("ChatPanel")
            chat_layout = QVBoxLayout(self.chat_panel)
            chat_layout.setContentsMargins(38, 22, 38, 26)
            chat_layout.setSpacing(18)

            top_row = QHBoxLayout()
            top_row.addStretch(1)
            self.mode_badge = QLabel()
            self.mode_badge.setObjectName("Subtle")
            top_row.addWidget(self.mode_badge)
            self.theme_hint = QLabel("\u2600")
            self.theme_hint.setObjectName("Subtle")
            self.theme_hint.setStyleSheet("font-size: 22px;")
            top_row.addWidget(self.theme_hint)
            self.more_hint = QLabel("\u22ef")
            self.more_hint.setObjectName("Subtle")
            self.more_hint.setStyleSheet("font-size: 22px;")
            top_row.addWidget(self.more_hint)
            chat_layout.addLayout(top_row)

            self.greeting_card = QFrame()
            self.greeting_card.setObjectName("GreetingCard")
            greeting_layout = QVBoxLayout(self.greeting_card)
            greeting_layout.setContentsMargins(28, 26, 28, 26)
            greeting_layout.setSpacing(8)
            self.greeting_title = QLabel()
            self.greeting_title.setObjectName("Headline")
            self.greeting_text = QLabel()
            self.greeting_text.setObjectName("Subtle")
            self.greeting_text.setStyleSheet("font-size: 16px;")
            greeting_layout.addWidget(self.greeting_title)
            greeting_layout.addWidget(self.greeting_text)
            chat_layout.addWidget(self.greeting_card)

            self.scroll_area = QScrollArea()
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setFrameShape(QFrame.NoFrame)
            self.messages_host = QWidget()
            self.messages_host.setObjectName("MessagesHost")
            self.messages_layout = QVBoxLayout(self.messages_host)
            self.messages_layout.setContentsMargins(0, 6, 0, 6)
            self.messages_layout.setSpacing(16)
            self.messages_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
            self.scroll_area.setWidget(self.messages_host)
            chat_layout.addWidget(self.scroll_area, 1)

            self.composer = QFrame()
            self.composer.setObjectName("Composer")
            composer_layout = QHBoxLayout(self.composer)
            composer_layout.setContentsMargins(14, 10, 10, 10)
            composer_layout.setSpacing(8)

            self.plus_button = QPushButton("")
            self.plus_button.setObjectName("RoundButton")
            self.plus_button.clicked.connect(self.show_plus_menu)
            composer_layout.addWidget(self.plus_button, 0, Qt.AlignVCenter)

            self.message_input = QPlainTextEdit()
            self.message_input.setObjectName("ComposerInput")
            self.message_input.setMinimumHeight(26)
            self.message_input.setMaximumHeight(120)
            self.message_input.setFrameShape(QFrame.NoFrame)
            self.message_input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.message_input.viewport().setAutoFillBackground(False)
            self.message_input.viewport().setStyleSheet("background: transparent;")
            composer_layout.addWidget(self.message_input, 1)

            self.micro_button = QPushButton("")
            self.micro_button.setObjectName("RoundButton")
            composer_layout.addWidget(self.micro_button, 0, Qt.AlignVCenter)

            self.send_button = QPushButton("↑")
            self.send_button.setObjectName("SendButton")
            self.send_button.clicked.connect(self.send_message)
            composer_layout.addWidget(self.send_button, 0, Qt.AlignVCenter)
            chat_layout.addWidget(self.composer)
            root.addWidget(self.chat_panel, 4)

        def apply_theme(self) -> None:
            app = QApplication.instance()
            if app is None:
                return
            self.effective_theme = "dark" if self.theme_mode == "system" else self.theme_mode
            app.setStyleSheet(DARK_STYLESHEET if self.effective_theme == "dark" else LIGHT_STYLESHEET)
            self.apply_theme_assets()
            self.refresh_history()

        def icon_color(self) -> str:
            return "#ECECF1" if self.effective_theme == "dark" else "#111827"

        def subtle_icon_color(self) -> str:
            return "#AAB0BC" if self.effective_theme == "dark" else "#6B7280"

        def apply_theme_assets(self) -> None:
            strong = self.icon_color()
            subtle = self.subtle_icon_color()
            self.sidebar_app_button.setIcon(themed_icon("sidebar_app.svg", strong, 28))
            self.new_chat_button.setIcon(themed_icon("new_chat.svg", strong, 22))
            self.search_compact_button.setIcon(themed_icon("search.svg", subtle, 22))
            self.settings_button.setIcon(themed_icon("settings.svg", strong, 22))
            self.search_icon.setPixmap(themed_pixmap("search.svg", subtle, 18))
            self.plus_button.setIcon(themed_icon("plus.svg", strong, 18))
            self.plus_button.setIconSize(QSize(18, 18))
            self.micro_button.setIcon(themed_icon("mic.svg", strong, 18))
            self.micro_button.setIconSize(QSize(18, 18))
            self.send_button.setText("\u2191")
            self.theme_hint.setText("\u263E" if self.effective_theme == "dark" else "\u2600")
            self.more_hint.setText("\u22EF")

        def retranslate_ui(self) -> None:
            self.new_chat_button.setText(tr(self.language, "new_chat"))
            self.search_input.setPlaceholderText(tr(self.language, "search"))
            self.history_title.setText(tr(self.language, "history"))
            self.settings_button.setText(tr(self.language, "settings"))
            self.greeting_title.setText(tr(self.language, "greeting_title"))
            self.greeting_text.setText(tr(self.language, "greeting_text"))
            self.message_input.setPlaceholderText(tr(self.language, "input_placeholder"))
            badge = f"{tr(self.language, 'mode_badge')}: {self.mode_label}"
            if self.model_label:
                badge = f"{badge} | {self.model_label}"
            self.mode_badge.setText(badge)
            self.update_sidebar_ui()
            self.refresh_history()
            self.render_messages()

        def toggle_sidebar(self) -> None:
            self.sidebar_expanded = not self.sidebar_expanded
            self.update_sidebar_ui()

        def focus_search(self) -> None:
            if not self.sidebar_expanded:
                self.sidebar_expanded = True
                self.update_sidebar_ui()
            self.search_input.setFocus()

        def update_sidebar_ui(self) -> None:
            expanded = self.sidebar_expanded
            self.sidebar.setFixedWidth(320 if expanded else 88)
            self.brand_text.setVisible(expanded)
            self.search_box.setVisible(expanded)
            self.search_compact_button.setVisible(not expanded)
            self.history_title.setVisible(expanded)
            self.history_panel.setVisible(expanded)
            self.sidebar.layout().setStretchFactor(self.history_panel, 1 if expanded else 0)
            self.sidebar.layout().setStretchFactor(self.sidebar_spacer, 0 if expanded else 1)

            for button in (self.new_chat_button, self.settings_button):
                button.setObjectName(
                    ("SidebarPrimaryButton" if button is self.new_chat_button else "SidebarFooterButton")
                    if expanded
                    else "SidebarCompactButton"
                )
                button.style().unpolish(button)
                button.style().polish(button)

            if expanded:
                self.new_chat_button.setText(tr(self.language, "new_chat"))
                self.settings_button.setText(tr(self.language, "settings"))
                self.search_compact_button.setToolTip("")
                self.new_chat_button.setToolTip("")
                self.settings_button.setToolTip("")
                self.new_chat_button.setIconSize(QSize(20, 20))
                self.settings_button.setIconSize(QSize(20, 20))
            else:
                self.new_chat_button.setText("")
                self.settings_button.setText("")
                self.search_compact_button.setToolTip(tr(self.language, "search"))
                self.new_chat_button.setToolTip(tr(self.language, "new_chat"))
                self.settings_button.setToolTip(tr(self.language, "settings"))
                self.new_chat_button.setIconSize(QSize(22, 22))
                self.settings_button.setIconSize(QSize(22, 22))

        def seed_default_conversations(self) -> None:
            day = tr(self.language, "days_ago")
            self.conversations = [
                Conversation(
                    title="Design chat AI interface",
                    subtitle=tr(self.language, "today"),
                    messages=[
                        ChatMessage(sender="ai", text=tr(self.language, "greeting_text")),
                    ],
                ),
                Conversation("Configure YOLO11 imgsz", tr(self.language, "yesterday")),
                Conversation("Choose YOLO11 version", f"2 {day}"),
                Conversation("YOLO config error", f"2 {day}"),
                Conversation("YOLO11 FPS RTX 3050 Ti", f"3 {day}"),
                Conversation("YOLO11 version and selection", f"3 {day}"),
                Conversation("Interface design request", f"4 {day}"),
                Conversation("AI health analysis", f"4 {day}"),
            ]
            self.active_conversation_index = 0
            self.refresh_history()
            self.history_list.setCurrentRow(0)
            self.render_messages()

        def active_conversation(self) -> Conversation:
            return self.conversations[self.active_conversation_index]

        def refresh_history(self) -> None:
            query = self.search_input.text().strip().lower() if hasattr(self, "search_input") else ""
            self.is_refreshing_history = True
            self.history_list.blockSignals(True)
            self.history_list.clear()
            current_item_row = 0
            visible_row = 0
            for index, conversation in enumerate(self.conversations):
                if query and query not in conversation.title.lower():
                    continue
                item = QListWidgetItem()
                item.setData(Qt.UserRole, index)
                item.setSizeHint(QSize(0, 68))
                self.history_list.addItem(item)
                widget = HistoryItemWidget(
                    conversation.title,
                    conversation.subtitle,
                    icon_color=self.subtle_icon_color(),
                    selected=index == self.active_conversation_index,
                )
                self.history_list.setItemWidget(item, widget)
                if index == self.active_conversation_index:
                    current_item_row = visible_row
                visible_row += 1
            self.history_list.blockSignals(False)
            if self.history_list.count():
                self.history_list.setCurrentRow(min(current_item_row, self.history_list.count() - 1))
            self.is_refreshing_history = False

        def select_conversation(self, row: int) -> None:
            if self.is_refreshing_history:
                return
            item = self.history_list.item(row)
            if item is None:
                return
            source_index = item.data(Qt.UserRole)
            if source_index is None:
                return
            self.active_conversation_index = int(source_index)
            self.render_messages()

        def render_messages(self) -> None:
            while self.messages_layout.count() > 1:
                item = self.messages_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            for message in self.active_conversation().messages:
                bubble = ChatBubble(message, language=self.language, align_right=message.sender == "user")
                self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
            QTimer.singleShot(
                0,
                lambda: self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum()
                ),
            )

        def start_new_chat(self) -> None:
            conversation = Conversation(
                title=tr(self.language, "new_chat"),
                subtitle=tr(self.language, "today"),
                messages=[ChatMessage(sender="ai", text=tr(self.language, "greeting_text"))],
            )
            self.conversations.insert(0, conversation)
            self.active_conversation_index = 0
            self.refresh_history()
            self.history_list.setCurrentRow(0)
            self.render_messages()

        def show_plus_menu(self) -> None:
            menu = QMenu(self)
            image_action = QAction(tr(self.language, "choose_image"), self)
            image_action.triggered.connect(self.pick_image)
            menu.addAction(image_action)

            text_action = QAction(tr(self.language, "choose_text"), self)
            text_action.triggered.connect(self.pick_text_file)
            menu.addAction(text_action)

            camera_action = QAction(tr(self.language, "camera"), self)
            camera_action.triggered.connect(self.open_camera)
            menu.addAction(camera_action)
            menu.exec(self.plus_button.mapToGlobal(self.plus_button.rect().bottomLeft()))

        def add_message(self, message: ChatMessage) -> None:
            conversation = self.active_conversation()
            conversation.messages.append(message)
            if conversation.title == tr(self.language, "new_chat") and message.sender == "user":
                first_line = message.text.strip().splitlines()[0] if message.text.strip() else ""
                conversation.title = (first_line[:28] or tr(self.language, "new_chat")).strip()
            conversation.subtitle = tr(self.language, "today")
            self.refresh_history()
            self.render_messages()

        def send_message(self) -> None:
            text = self.message_input.toPlainText().strip()
            if not text:
                QMessageBox.information(self, tr(self.language, "info_title"), tr(self.language, "empty_send"))
                return
            self.add_message(ChatMessage(sender="user", text=text))
            self.message_input.clear()
            self.add_message(ChatMessage(sender="ai", text=self.build_ai_reply(text=text, source="text")))

        def build_ai_reply(self, *, text: str, source: str) -> str:
            if source == "image":
                return tr(self.language, "ai_reply_image")
            if source == "camera":
                return tr(self.language, "ai_reply_camera")
            if source == "text_file":
                return tr(self.language, "ai_reply_text")
            return f"{tr(self.language, 'ai_reply_text')} {text[:120]}".strip()

        def pick_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                tr(self.language, "choose_image"),
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
            )
            if not path:
                return
            self.add_message(
                ChatMessage(
                    sender="user",
                    text=tr(self.language, "attach_image_label"),
                    attachment_path=path,
                    attachment_kind="image",
                )
            )
            self.add_message(ChatMessage(sender="ai", text=self.build_ai_reply(text=path, source="image")))

        def pick_text_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                tr(self.language, "choose_text"),
                "",
                "Text files (*.txt)",
            )
            if not path:
                return
            preview_text = Path(path).read_text(encoding="utf-8", errors="ignore")[:1200].strip()
            message_text = f"{tr(self.language, 'txt_title')}:\n{preview_text or '(empty)'}"
            self.add_message(
                ChatMessage(
                    sender="user",
                    text=message_text,
                    attachment_path=path,
                    attachment_kind="text",
                )
            )
            self.add_message(ChatMessage(sender="ai", text=self.build_ai_reply(text=preview_text, source="text_file")))

        def open_camera(self) -> None:
            dialog = CameraCaptureDialog(
                language=self.language,
                camera_index_value=self.initial_camera_index,
                parent=self,
            )
            dialog.captured.connect(self.handle_camera_capture)
            dialog.exec()

        def handle_camera_capture(self, path: str) -> None:
            self.add_message(
                ChatMessage(
                    sender="user",
                    text=tr(self.language, "attach_camera_label"),
                    attachment_path=path,
                    attachment_kind="camera",
                )
            )
            self.add_message(ChatMessage(sender="ai", text=self.build_ai_reply(text=path, source="camera")))

        def open_settings(self) -> None:
            SettingsDialog(parent_window=self).exec()

    app = QApplication.instance() or QApplication(sys.argv)
    window = ChatWindow(title=window_title, initial_camera_index=camera_index, mode_label=app_mode, model_label=selected_model)
    window.show()
    return app.exec()
