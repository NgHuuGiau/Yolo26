from __future__ import annotations

import argparse
import math
import re
import sys
import json
import tempfile
import time
import random
import sqlite3
from dataclasses import dataclass, field
import os
import platform
from pathlib import Path
from typing import Literal

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import HtmlFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None

@dataclass
class ChatMessage:
    sender: Literal["user", "ai"]
    text: str
    attachment_path: str | None = None
    attachment_kind: Literal["image", "text", "camera"] | None = None
    id: int | None = None


@dataclass
class Conversation:
    title: str
    subtitle: str
    messages: list[ChatMessage] = field(default_factory=list)
    id: int | None = None


LEGACY_SEEDED_CONVERSATION_TITLES = {
    "Design chat AI interface",
    "Configure YOLO11 imgsz",
    "Choose YOLO11 version",
    "YOLO config error",
    "YOLO11 FPS RTX 3050 Ti",
    "YOLO11 version and selection",
    "Interface design request",
    "AI health analysis",
}


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
        from PySide6.QtCore import QByteArray, QSize, QTimer, Qt, Signal, QThread, QVariantAnimation, QEasingCurve, QTemporaryFile, QRectF, QPropertyAnimation, QPoint, QParallelAnimationGroup
        from PySide6.QtGui import QAction, QCloseEvent, QIcon, QImage, QPainter, QPixmap, QColor, QPen, QShortcut, QKeySequence, QPalette, QTextOption
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
            QToolTip,
            QSystemTrayIcon,
            QPushButton,
            QProgressBar,
            QPlainTextEdit,
            QScrollArea,
            QSizePolicy,
            QSpacerItem,
            QGraphicsDropShadowEffect,
            QGraphicsBlurEffect,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("Missing PySide6. Install with: pip install PySide6 opencv-python") from exc

    try:
        import numpy as np
        import pyaudio
        import wave
        import torch
        from faster_whisper import WhisperModel
        VOICE_AI_AVAILABLE = True
    except ImportError:
        VOICE_AI_AVAILABLE = False

    voice_model_cache: dict[tuple[str, str], object] = {}

    def get_cached_whisper_model(*, language: str, device: str):
        cache_key = (language, device)
        model = voice_model_cache.get(cache_key)
        if model is None:
            compute_type = "int8" if device == "cpu" else "float16"
            model = WhisperModel("base", device=device, compute_type=compute_type)
            voice_model_cache[cache_key] = model
        return model

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
            "greeting_title": "Hello!",
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
            "image_model_error": "AI chat is unavailable in this build.",
            "ai_unavailable": "AI chat has been removed from this build.",
            "ai_reply_camera": "Camera snapshot received and added to the chat.",
            "empty_send": "Enter a message before sending.",
            "info_title": "Info",
            "loading_model": "Waking up Voice AI...",
            "recording_status": "Listening...",
            "voice_error": "Could not recognize voice or Mic error.",
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
            "delete_chat": "Delete chat",
            "confirm_delete_title": "Confirm Delete",
            "confirm_delete_msg": "Are you sure you want to delete this conversation?",
        },
        "vi": {
            "copied_hint": "Đã sao chép vào bộ nhớ tạm!",
            "new_message": "Tin nhắn mới từ AI",
            "new_chat": "Chat mới",
            "search": "Tìm kiếm đoạn chat",
            "history": "Lịch sử chat",
            "settings": "Cài đặt",
            "greeting_title": "Chào bạn!",
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
            "light": "Sáng",
            "capture": "Chụp",
            "camera_ready": "Camera đã sẵn sàng. Nhấn Chụp để thêm ảnh vào đoạn chat.",
            "camera_unavailable": "Không mở được camera.",
            "camera_missing": "Chưa cài opencv-python.",
            "txt_title": "Xem trước tệp",
            "ai_reply_text": "Tôi đã nhận nội dung và sẽ xử lý trong ngữ cảnh đoạn chat này.",
            "ai_reply_image": "Đã nhận ảnh. Tôi sẽ dùng ảnh làm dữ liệu đầu vào cho đoạn chat này.",
            "image_model_error": "Tính năng chat AI không có trong bản dựng này.",
            "ai_unavailable": "Tính năng chat AI đã được gỡ khỏi bản dựng này.",
            "ai_reply_camera": "Đã nhận ảnh chụp từ camera và thêm vào đoạn chat.",
            "empty_send": "Hãy nhập tin nhắn trước khi gửi.",
            "info_title": "Thông tin",
            "loading_model": "Đang nạp AI giọng nói...",
            "recording_status": "Đang lắng nghe...",
            "voice_error": "Không nhận diện được giọng nói hoặc lỗi mic.",
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
            "delete_chat": "Xóa đoạn chat",
            "confirm_delete_title": "Xác nhận xóa",
            "confirm_delete_msg": "Bạn có chắc chắn muốn xóa đoạn chat này không?",
        },
    }

    DARK_STYLESHEET = """
    QMainWindow, QDialog, QWidget#Root {
        background: #0b0b0b;
        font-family: "Segoe UI", "Arial", sans-serif;
        font-family: "Inter", "Segoe UI", "Roboto", "Arial", sans-serif;
    }
    QWidget#ChatPanel,
    QWidget#ScrollContainer,
    QWidget#MessagesHost {
        background: transparent;
    }
    QLabel {
        background: transparent;
        color: #e3e3e3;
        font-weight: normal;
    }
    QLabel#Subtle {
        color: #b4b4b4;
    }
    QLabel#GreetingTitle {
        font-size: 34px;
        font-weight: 700;
        color: #ffffff;
    }
    QLabel#BrandText {
        font-size: 22px;
        font-weight: 700;
        color: #e3e3e3;
    }
    QLabel#ChatHeaderTitle {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }
    QLabel#SectionTitle {
        color: #ffffff;
        font-size: 15px;
        font-weight: 600;
    }
    QLabel#ModeBadge {
        color: #c7c7cc;
        font-size: 13px;
    }
    QLabel#GreetingAvatar {
        min-width: 56px;
        max-width: 56px;
        min-height: 56px;
        max-height: 56px;
        border-radius: 28px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
    }
    QLabel#Attachment {
        color: #d7e0ff;
        font-weight: 600;
    }
    QFrame#Sidebar {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 rgba(18, 19, 24, 0.98),
            stop:1 rgba(13, 14, 18, 0.98));
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        border-top-right-radius: 30px;
        border-bottom-right-radius: 30px;
    }
    QFrame#TopBar {
        background: transparent;
        border: none;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    QFrame#SidebarHeader {
        background: transparent;
        border: none;
    }
    QFrame#SidebarButton,
    QFrame#GreetingCard,
    QFrame#ChatBoard,
    QFrame#SettingsCard {
        background: transparent;
        border: none;
        border-radius: 20px;
    }
    QFrame#Composer {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 rgba(30, 32, 39, 0.99),
            stop:1 rgba(18, 20, 26, 0.99));
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 30px;
    }
    QFrame#ComposerInputRow,
    QWidget#ComposerPreviewHost {
        background: transparent;
        border: none;
    }
    QFrame#ComposerInputRow {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 24px;
        padding: 2px 4px;
    }
    QScrollArea#ComposerPreviewScroll {
        background: transparent;
        border: none;
    }
    QFrame#ComposerPreviewThumb {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
    }
    QPushButton#ComposerPreviewDeleteButton {
        background: rgba(239, 68, 68, 0.92);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 700;
        padding: 0;
        text-align: center;
    }
    QPushButton#ComposerPreviewDeleteButton:hover {
        background: rgba(220, 38, 38, 0.98);
    }
    QFrame#HistoryPanel {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
    }
    QFrame#SearchBox {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
    }
    QFrame#MessageScroll {
        background: transparent;
        border: none;
    }
    QFrame#SettingsShell,
    QFrame#ImagePreviewShell { /* Thêm ID cho shell của ImagePreviewDialog */
        background: #111111;
        border: 1px solid #3a3a3a;
        border-radius: 28px;
    }
    QFrame#SettingsNav {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 22px;
    }
    QFrame#SettingsContent {
        background: transparent;
        border: none;
        border-radius: 22px;
    }
    QFrame#SettingsOptionCard {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
    }
    QFrame#HistoryItem {
        background: transparent;
        border: none;
        border-radius: 16px;
    }
    QFrame#HistoryItem:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    QFrame#HistoryItem[selected="true"] {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    QFrame#BubbleUser {
        background: rgba(77, 184, 255, 0.14);
        border: 1px solid rgba(77, 184, 255, 0.12);
        border-radius: 34px;
    }
    QFrame#BubbleAI {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 34px;
    }
    QLabel#Avatar {
        min-width: 32px; /* Nhỏ hơn */
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        border-radius: 16px; /* Bo tròn */
        border: none;
        background: #4db8ff; /* Accent blue */
        color: white;
        font-weight: 700;
        qproperty-alignment: AlignCenter;
    }
    QLineEdit, QPlainTextEdit, QComboBox, QListWidget {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        color: #e3e3e3; /* Màu chữ */
        padding: 12px 14px;
    }
    QLineEdit[state="error"] { border: 2px solid #FF5252; }
    QLineEdit[state="success"] { border: 2px solid #4db8ff; }
    QLineEdit[state="error"] { border: 1.5px solid #FF5252; }
    QLineEdit[state="success"] { border: 1.5px solid #4CAF50; }
    QScrollArea#MessageScroll {
        border: none;
        background: transparent;
    }
    QFrame#Composer QPlainTextEdit {
        background: transparent;
        border: none;
        border-radius: 0;
        color: #f5f7fb;
        font-size: 16px;
        padding: 12px 8px 12px 8px;
        min-height: 36px;
        max-height: 88px;
    }
    QPlainTextEdit#ComposerInput,
    QPlainTextEdit#ComposerInput QWidget { /* Đảm bảo viewport trong suốt */
        background: transparent;
        border: none;
    }
    QPlainTextEdit {
        padding-top: 0px;
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
        background: transparent;
        color: #ffffff;
        border: none;
        border-radius: 18px;
        padding: 10px 14px;
        text-align: left;
    }
    QPushButton:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    QPushButton#SidebarPrimaryButton {
        min-height: 54px;
        padding-left: 16px;
        font-size: 15px;
        font-weight: 600;
        border: none;
    }
    QPushButton#SidebarPrimaryButton {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
    }
    QPushButton#TopActionButton {
        min-width: 40px;
        max-width: 40px;
        min-height: 40px;
        max-height: 40px;
        padding: 0;
        border-radius: 20px;
        background: transparent;
        color: #e3e3e3;
        font-size: 18px;
        text-align: center;
    }
    QPushButton#TopActionButton:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    QPushButton#SidebarPrimaryButton:hover,
    QPushButton#SidebarCompactButton:hover,
    QPushButton#SidebarCompactSearchButton:hover,
    QFrame#SearchBox:hover {
        background: rgba(255, 255, 255, 0.07);
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
        border-radius: 20px;
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
        border-radius: 20px;
        text-align: center;
        border: none;
        background: transparent;
    }
    QPushButton#RoundButton {
        min-width: 44px;
        max-width: 44px;
        min-height: 44px;
        max-height: 44px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        background: rgba(255, 255, 255, 0.04);
        color: #f2f4f8;
        font-size: 20px;
        padding: 0;
        text-align: center;
    }
    QPushButton#RoundButton:hover {
        background: rgba(255, 255, 255, 0.12);
    }
    QPushButton#SendButton {
        min-width: 44px;
        max-width: 44px;
        min-height: 44px;
        max-height: 44px;
        border-radius: 20px;
        border: none;
        background: #4db8ff; /* Accent blue */
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
    }
    QPushButton#SendButton:hover {
        background: #86cbff; /* Hover sáng hơn */
    }
    QPushButton#SendButton {
        background: #2ea8ff;
    }
    QPushButton#SendButton:hover {
        background: #62c0ff;
    }
    QPushButton#SendButton {
        background: #1f9cff;
    }
    QPushButton#SendButton:hover {
        background: #58b9ff;
    }
    QPushButton#SettingsNavButton {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        text-align: left;
        padding: 14px 18px;
        font-weight: 600;
        color: #f3f4f6;
    }
    QPushButton#SettingsCloseButton {
        background: transparent;
        border: none;
        color: #e3e3e3;
        font-size: 18px;
        padding: 0;
    }
    QPushButton#SettingsCloseButton:hover {
        color: #e3e3e3; /* Giữ nguyên, không đổi hover */
        background: transparent;
    }
    QComboBox#SettingsCombo {
        min-width: 220px;
        min-height: 44px;
        padding: 10px 14px;
        font-size: 15px;
        font-weight: 600;
        color: #f3f4f6;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
    }
    QComboBox#SettingsCombo::drop-down {
        border: none;
        width: 0px;
    }
    QComboBox#SettingsCombo::down-arrow {
        image: none;
    }
    QComboBox#SettingsCombo QAbstractItemView {
        background: #1c1d22;
        color: #f3f4f6;
        selection-background-color: rgba(77, 184, 255, 0.2);
        selection-color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.08);
        outline: none;
    }
    QMenu {
        background: #1b1c20;
        border: 1px solid #3a3a3a;
        border-radius: 16px;
        padding: 8px;
    }
    QMenu::item {
        padding: 10px 18px;
        border-radius: 10px;
        color: #ffffff;
    }
    QMenu::item:selected {
        background: #242424;
    }
    """

    LIGHT_STYLESHEET = """
    QMainWindow, QDialog, QWidget#Root {
        background: #f7f7f8;
        font-family: "Segoe UI", "Arial", sans-serif;
        font-family: "Inter", "Segoe UI", "Roboto", "Arial", sans-serif;
    }
    QWidget#ChatPanel,
    QWidget#ScrollContainer,
    QWidget#MessagesHost {
        background: transparent;
    }
    QLabel {
        background: transparent;
        color: #111827;
        font-weight: normal;
    }
    QLabel#Subtle {
        color: #374151;
    }
    QLabel#GreetingTitle {
        font-size: 34px;
        font-weight: 700;
        color: #000000;
    }
    QLabel#BrandText {
        font-size: 26px;
        font-weight: 700;
        color: #000000;
    }
    QLabel#ChatHeaderTitle {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
    }
    QLabel#SectionTitle {
        color: #000000;
        font-size: 15px;
        font-weight: 600;
    }
    QLabel#ModeBadge {
        color: #6b7280;
        font-size: 13px;
    }
    QLabel#GreetingAvatar {
        min-width: 56px;
        max-width: 56px;
        min-height: 56px;
        max-height: 56px;
        border-radius: 28px;
        border: 1px solid rgba(17, 24, 39, 0.08);
        background: rgba(17, 24, 39, 0.03);
    }
    QLabel#Attachment {
        color: #000000;
        font-weight: 600;
    }
    QFrame#Sidebar {
        background: rgba(255, 255, 255, 0.92);
        border-right: 1px solid rgba(17, 24, 39, 0.08);
        border-top-right-radius: 30px;
        border-bottom-right-radius: 30px;
    }
    QFrame#TopBar {
        background: transparent;
        border: none;
        border-bottom: 1px solid rgba(17, 24, 39, 0.08);
    }
    QFrame#SidebarHeader {
        background: transparent;
        border: none;
    }
    QFrame#SidebarButton,
    QFrame#GreetingCard, 
    QFrame#ChatBoard,
    QFrame#SettingsCard {
        background: transparent;
        border: none;
        border-radius: 20px;
    }
    QFrame#Composer {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid rgba(17, 24, 39, 0.10);
        border-radius: 30px;
    }
    QFrame#ComposerInputRow,
    QWidget#ComposerPreviewHost {
        background: transparent;
        border: none;
    }
    QFrame#ComposerInputRow {
        background: rgba(17, 24, 39, 0.035);
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 24px;
        padding: 2px 4px;
    }
    QScrollArea#ComposerPreviewScroll {
        background: transparent;
        border: none;
    }
    QFrame#ComposerPreviewThumb {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 18px;
    }
    QPushButton#ComposerPreviewDeleteButton {
        background: rgba(239, 68, 68, 0.92);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 700;
        padding: 0;
        text-align: center;
    }
    QPushButton#ComposerPreviewDeleteButton:hover {
        background: rgba(220, 38, 38, 0.98);
    }
    QFrame#HistoryPanel {
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid rgba(17, 24, 39, 0.06);
        border-radius: 24px;
    }
    QFrame#SearchBox {
        background: rgba(17, 24, 39, 0.03);
        border: 1px solid rgba(17, 24, 39, 0.06);
        border-radius: 20px;
    }
    QFrame#MessageScroll {
        background: transparent;
        border: none;
    }
    QFrame#SettingsShell,
    QFrame#ImagePreviewShell { /* Thêm ID cho shell của ImagePreviewDialog */
        background: #ffffff;
        border: 1px solid #d9d9df;
        border-radius: 28px;
    }
    QFrame#SettingsNav {
        background: rgba(17, 24, 39, 0.03);
        border: 1px solid rgba(17, 24, 39, 0.06);
        border-radius: 22px;
    }
    QFrame#SettingsContent {
        background: transparent;
        border: none;
        border-radius: 22px;
    }
    QFrame#SettingsOptionCard {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 20px;
    }
    QFrame#HistoryItem {
        background: transparent;
        border: none;
        border-radius: 16px;
    }
    QFrame#HistoryItem:hover {
        background: rgba(17, 24, 39, 0.05);
    }
    QFrame#HistoryItem[selected="true"] {
        background: rgba(17, 24, 39, 0.08);
        border: 1px solid rgba(17, 24, 39, 0.05);
    }
    QFrame#BubbleUser {
        background: rgba(77, 184, 255, 0.12);
        border: 1px solid rgba(77, 184, 255, 0.12);
        border-radius: 34px;
    }
    QFrame#BubbleAI {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(17, 24, 39, 0.06);
        border-radius: 34px;
    }
    QLabel#Avatar {
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        border-radius: 16px;
        border: none;
        background: #4db8ff;
        color: #111827;
        font-weight: 700;
        qproperty-alignment: AlignCenter;
    }
    QLineEdit, QPlainTextEdit, QComboBox, QListWidget {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 16px;
        color: #111827;
        padding: 12px 14px;
    }
    QLineEdit[state="error"] { border: 2px solid #FF5252; }
    QLineEdit[state="success"] { border: 2px solid #4db8ff; }
    QLineEdit[state="error"] { border: 1.5px solid #FF5252; }
    QLineEdit[state="success"] { border: 1.5px solid #4CAF50; }
    QScrollArea#MessageScroll {
        border: none;
        background: transparent;
    }
    QFrame#Composer QPlainTextEdit {
        background: transparent;
        border: none;
        border-radius: 0;
        color: #111827;
        font-size: 16px;
        padding: 12px 8px 12px 8px;
        min-height: 36px;
        max-height: 88px;
    }
    QPlainTextEdit#ComposerInput,
    QPlainTextEdit#ComposerInput QWidget { /* Đảm bảo viewport trong suốt */
        background: transparent;
        border: none;
    }
    QPlainTextEdit {
        padding-top: 0px;
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
        background: transparent;
        color: #111827;
        border: none;
        border-radius: 18px;
        padding: 10px 14px;
        text-align: left;
    }
    QPushButton:hover {
        background: rgba(17, 24, 39, 0.05);
    }
    QPushButton#SidebarPrimaryButton {
        min-height: 54px;
        padding-left: 16px;
        font-size: 15px;
        font-weight: 600;
        border: none;
    }
    QPushButton#SidebarPrimaryButton {
        background: transparent;
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 16px;
    }
    QPushButton#TopActionButton {
        min-width: 40px;
        max-width: 40px;
        min-height: 40px;
        max-height: 40px;
        padding: 0;
        border-radius: 20px;
        background: transparent;
        color: #111827;
        font-size: 18px;
        text-align: center;
    }
    QPushButton#TopActionButton:hover {
        background: rgba(17, 24, 39, 0.08);
    }
    QPushButton#SidebarPrimaryButton:hover,
    QPushButton#SidebarCompactButton:hover,
    QPushButton#SidebarCompactSearchButton:hover,
    QFrame#SearchBox:hover {
        background: rgba(17, 24, 39, 0.06);
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
        border-radius: 20px;
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
        border-radius: 20px;
        text-align: center;
        border: none;
        background: transparent;
    }
    QPushButton#RoundButton {
        min-width: 44px;
        max-width: 44px;
        min-height: 44px;
        max-height: 44px;
        border-radius: 20px;
        border: 1px solid rgba(17, 24, 39, 0.08);
        background: rgba(255, 255, 255, 0.92);
        color: #111827;
        font-size: 20px;
        padding: 0;
        text-align: center;
    }
    QPushButton#RoundButton:hover {
        background: rgba(17, 24, 39, 0.08);
    }
    QPushButton#SendButton {
        min-width: 44px;
        max-width: 44px;
        min-height: 44px;
        max-height: 44px;
        border-radius: 20px;
        border: none;
        background: #4db8ff;
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
    }
    QPushButton#SendButton:hover {
        background: #86cbff;
    }
    QPushButton#SettingsNavButton {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 16px;
        text-align: left;
        padding: 14px 18px;
        font-weight: 600;
    }
    QPushButton#SettingsCloseButton {
        background: transparent;
        border: none;
        color: #111827;
        font-size: 18px;
        padding: 0;
    }
    QPushButton#SettingsCloseButton:hover {
        color: #111827; /* Không hover đổi màu/nền */
        background: transparent;
    }
    QComboBox#SettingsCombo {
        min-width: 220px;
        min-height: 44px;
        padding: 10px 14px;
        font-size: 15px;
        font-weight: 600;
        color: #000000;
        background: #ffffff;
    }
    QComboBox#SettingsCombo::drop-down {
        border: none;
        width: 0px;
    }
    QComboBox#SettingsCombo::down-arrow {
        image: none;
    }
    QComboBox#SettingsCombo QAbstractItemView {
        background: #ffffff;
        color: #000000;
        selection-background-color: #dbeafe;
        selection-color: #000000;
        border: 1px solid #d9d9df;
        outline: none;
    }
    QMenu {
        background: #ffffff;
        border: 1px solid #d9d9df;
        border-radius: 16px;
        padding: 8px;
    }
    QMenu::item {
        padding: 10px 18px;
        border-radius: 10px;
        color: #111827;
    }
    QMenu::item:selected {
        background: #f7f7f8;
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
            self.frame_count += 1
            now = time.monotonic()
            if self.last_fps_time == 0.0:
                self.last_fps_time = now
            elapsed = now - self.last_fps_time
            if elapsed >= 0.5:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_time = now
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

    class TypingIndicator(QWidget):
        def __init__(self, color: QColor, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setFixedSize(60, 24)
            self.color = color
            self.start_time = time.time()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update)
            self.timer.start(30)

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            t = time.time() - self.start_time
            for i in range(3):
                offset = math.sin(t * 7 - i * 0.9) * 4
                alpha = int(160 + 95 * math.sin(t * 7 - i * 0.9))
                c = QColor(self.color)
                c.setAlpha(max(0, min(255, alpha)))
                painter.setBrush(c)
                painter.drawEllipse(10 + i * 14, 12 + offset, 5, 5)

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

            divider = QFrame()
            divider.setFrameShape(QFrame.VLine)
            divider.setStyleSheet(
                "background: transparent; border: none; border-left: 1px solid rgba(255, 255, 255, 0.08);"
                if self.window.effective_theme == "dark"
                else "background: transparent; border: none; border-left: 1px solid rgba(17, 24, 39, 0.08);"
            )
            body_row.addWidget(divider)

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
            self.setCursor(Qt.PointingHandCursor)
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

    class ChatDatabase:
        def __init__(self, db_path: str):
            self.db_path = db_path
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._init_db()

        def _init_db(self):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT,
                        subtitle TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id INTEGER,
                        sender TEXT,
                        text TEXT,
                        attachment_path TEXT,
                        attachment_kind TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)

        def get_setting(self, key: str, default: str) -> str:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
                    row = cursor.fetchone()
                    return row[0] if row else default
            except Exception:
                return default

        def set_setting(self, key: str, value: str):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))

        def get_all_conversations(self) -> list[Conversation]:
            convs = []
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT id, title, subtitle FROM conversations ORDER BY id DESC")
                    for row in cursor:
                        c_id, title, subtitle = row
                        messages = []
                        m_cursor = conn.execute(
                            "SELECT sender, text, attachment_path, attachment_kind, id FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                            (c_id,)
                        )
                        for m_row in m_cursor:
                            sender, text, path, kind, m_id = m_row
                            messages.append(ChatMessage(sender=sender, text=text, attachment_path=path, attachment_kind=kind, id=m_id))
                        convs.append(Conversation(title=title, subtitle=subtitle, messages=messages, id=c_id))
            except Exception as e:
                print(f"Database error: {e}")
            return convs

        def create_conversation(self, title: str, subtitle: str) -> int:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("INSERT INTO conversations (title, subtitle) VALUES (?, ?)", (title, subtitle))
                return cursor.lastrowid

        def update_conversation_title(self, conv_id: int, title: str):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))

        def add_message(self, conv_id: int, msg: ChatMessage) -> int:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO messages (conversation_id, sender, text, attachment_path, attachment_kind) VALUES (?, ?, ?, ?, ?)",
                    (conv_id, msg.sender, msg.text, msg.attachment_path, msg.attachment_kind)
                )
                return cursor.lastrowid

        def delete_conversation(self, conv_id: int):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

        def delete_conversations_by_titles(self, titles: set[str]) -> None:
            if not titles:
                return
            placeholders = ",".join("?" for _ in titles)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"DELETE FROM conversations WHERE title IN ({placeholders})",
                    tuple(titles),
                )

        def clear_all_conversations(self):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM conversations")

        def clear_all_messages(self):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM messages")

        def search_conversations_by_message(self, query: str) -> list[int]:
            if not query:
                return []
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT DISTINCT conversation_id FROM messages WHERE text LIKE ?", (f"%{query}%",))
                    return [row[0] for row in cursor]
            except Exception:
                return []

    class WaveformWidget(QWidget):
        def __init__(self, color: str, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setFixedSize(100, 30)
            self.values = [0] * 12
            self.color = QColor(color)

        def set_intensity(self, value: int) -> None:
            self.values.pop(0)
            self.values.append(value)
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            spacing = 4
            bar_w = (w - (len(self.values) - 1) * spacing) / len(self.values)
            for i, val in enumerate(self.values):
                bar_h = max(4, (val / 100.0) * h)
                x = i * (bar_w + spacing)
                y = (h - bar_h) / 2
                painter.setBrush(self.color)
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(QRectF(x, y, bar_w, bar_h), 2, 2)

    class RecordingPanel(QFrame):
        def __init__(self, language: str, parent: QWidget | None = None, window: QMainWindow | None = None) -> None:
            super().__init__(parent)
            self.setObjectName("RecordingPanel")
            self._window = window
            self.setGraphicsEffect(QGraphicsDropShadowEffect(blurRadius=15, xOffset=0, yOffset=4, color=QColor(0,0,0,60)))
            layout = QHBoxLayout(self)
            layout.setContentsMargins(16, 8, 16, 8)
            layout.setSpacing(12)
            
            self.waveform = WaveformWidget("#FF5252")
            layout.addWidget(self.waveform)
            
            self.label = QLabel(tr(language, "recording_status"))
            self.label.setStyleSheet("color: #FF5252; font-weight: 700; font-size: 14px;")
            layout.addWidget(self.label)
            self.hide()

        def setup_styles(self) -> None:
            if self._window and getattr(self._window, "effective_theme", "dark") == "light":
                self.setStyleSheet("background: rgba(0,0,0,0.55); border: none; border-radius: 16px;")
                self.label.setStyleSheet("color: #111827; font-weight: 700; font-size: 14px;")
            else:
                self.setStyleSheet("background: rgba(0,0,0,0.6); border: none; border-radius: 16px;")
                self.label.setStyleSheet("color: white; font-weight: 700; font-size: 14px;")

    if VOICE_AI_AVAILABLE:
        class VoiceWorker(QThread):
            result_ready = Signal(str)
            intensity_changed = Signal(int)
            finished = Signal()
            error = Signal()

            def __init__(self, language: str):
                super().__init__()
                self.lang_code = "vi" if language == "vi" else "en"
                self.is_running = True

            def run(self):
                import tempfile
                chunk = 1024
                fs = 16000
                p = pyaudio.PyAudio()
                stream = None
                tmp_path = None
                try:
                    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs,
                                    frames_per_buffer=chunk, input=True)
                    frames = []
                    silent_chunks = 0
                    while self.is_running:
                        data = stream.read(chunk, exception_on_overflow=False)
                        frames.append(data)
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
                        intensity = int(min(100, (rms ** 0.65) * 2.2))
                        self.intensity_changed.emit(intensity)
                        if rms < 80: silent_chunks += 1
                        else: silent_chunks = 0
                        if silent_chunks > int(fs / chunk * 1.8): break
                        if len(frames) > int(fs / chunk * 12): break
                    if stream and stream.is_active():
                        stream.stop_stream()
                    if stream:
                        stream.close()
                    if frames:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        tmp_path = tmp.name
                        tmp.close()
                        wf = wave.open(tmp_path, 'wb')
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(fs)
                        wf.writeframes(b''.join(frames))
                        wf.close()
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model = get_cached_whisper_model(language=self.lang_code, device=device)
                        segments, _ = model.transcribe(tmp_path, language=self.lang_code)
                        text = "".join([s.text for s in segments])
                        self.result_ready.emit(text.strip())
                    else:
                        self.result_ready.emit("")
                except Exception:
                    self.error.emit()
                finally:
                    try:
                        p.terminate()
                    except:
                        pass
                    self.finished.emit()
                    if tmp_path:
                        try: os.unlink(tmp_path)
                        except: pass

            def stop(self):
                self.is_running = False

    class ImagePreviewDialog(QDialog):
        def __init__(self, image_path: str, parent: QWidget | None = None, effective_theme: str = "dark") -> None:
            super().__init__(parent)
            self.setWindowTitle("Image Preview")
            self.setModal(True)
            self.resize(1000, 800)
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setWindowOpacity(0.0)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(10, 10, 10, 10)
            shell = QFrame()
            shell.setObjectName("ImagePreviewShell")
            shell_layout = QVBoxLayout(shell)
            shell_layout.setContentsMargins(20, 20, 20, 20)
            layout.addWidget(shell)

            self.blur = QGraphicsBlurEffect()
            self.blur.setBlurRadius(20)
            shell.setGraphicsEffect(self.blur)

            if effective_theme == "light":
                shell.setStyleSheet("background: rgba(255,255,255,0.85); border-radius: 28px; border: 1px solid rgba(0,0,0,0.08);")
            else:
                shell.setStyleSheet("background: rgba(18,19,24,0.92); border-radius: 28px; border: 1px solid rgba(255,255,255,0.06);")
            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setAlignment(Qt.AlignCenter)
            self.scroll.setStyleSheet("background: transparent; border: none; border-radius: 20px;")
            self.img_label = QLabel()
            self.img_label.setPixmap(QPixmap(image_path).scaled(960, 740, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.scroll.setWidget(self.img_label)
            shell_layout.addWidget(self.scroll)
            close_btn = QPushButton("✕", shell)
            close_btn.setObjectName("SettingsCloseButton")
            close_btn.setFixedSize(40, 40)
            close_btn.move(940, 10)
            close_btn.clicked.connect(self.accept)

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self.accept()
            else:
                super().keyPressEvent(event)

        def showEvent(self, event):
            self.group = QParallelAnimationGroup(self)
            self.fade = QPropertyAnimation(self, b"windowOpacity")
            self.fade.setDuration(300)
            self.fade.setStartValue(0.0)
            self.fade.setEndValue(1.0)
            
            self.blur_anim = QPropertyAnimation(self.blur, b"blurRadius")
            self.blur_anim.setDuration(500)
            self.blur_anim.setStartValue(20)
            self.blur_anim.setEndValue(0)
            
            self.group.addAnimation(self.fade)
            self.group.addAnimation(self.blur_anim)
            self.group.start()
            super().showEvent(event)

    class MessageInput(QPlainTextEdit):
        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.textChanged.connect(self._adjust_height)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
            self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
            self.setCenterOnScroll(False)
            self.setCursorWidth(3)
            self.document().setDocumentMargin(4)
            self.setContentsMargins(0, 0, 0, 0)
            self.setTabChangesFocus(False)
            self.setFocusPolicy(Qt.StrongFocus)
            self.setPlaceholderText("Nhập tin nhắn...")
            self._adjust_height()

        def apply_visual_style(self, *, dark_mode: bool) -> None:
            palette = self.palette()
            if dark_mode:
                palette.setColor(QPalette.Text, QColor("#F3F4F6"))
                palette.setColor(QPalette.PlaceholderText, QColor("#C2CAD6"))
                palette.setColor(QPalette.Base, QColor(0, 0, 0, 0))
                palette.setColor(QPalette.Highlight, QColor("#4DB8FF"))
                palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
                self.setStyleSheet(
                    "background: transparent;"
                    "border: none;"
                    "color: #F3F4F6;"
                    "selection-background-color: #4DB8FF;"
                    "selection-color: #FFFFFF;"
                    "font-size: 16px;"
                    "padding: 8px 6px;"
                )
            else:
                palette.setColor(QPalette.Text, QColor("#111827"))
                palette.setColor(QPalette.PlaceholderText, QColor("#6B7280"))
                palette.setColor(QPalette.Base, QColor(0, 0, 0, 0))
                palette.setColor(QPalette.Highlight, QColor("#93C5FD"))
                palette.setColor(QPalette.HighlightedText, QColor("#111827"))
                self.setStyleSheet(
                    "background: transparent;"
                    "border: none;"
                    "color: #111827;"
                    "selection-background-color: #93C5FD;"
                    "selection-color: #111827;"
                    "font-size: 16px;"
                    "padding: 8px 6px;"
                )
            self.setPalette(palette)
            self.viewport().setPalette(palette)
            self.viewport().setStyleSheet("background: transparent;")

        def _adjust_height(self):
            self.document().setTextWidth(self.viewport().width())
            height = (
                self.document().size().height()
                + self.contentsMargins().top()
                + self.contentsMargins().bottom()
                + 16
            )
            self.setFixedHeight(max(40, min(int(height), 88)))

        enter_pressed = Signal()

        def keyPressEvent(self, event):
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() & Qt.ShiftModifier:
                    super().keyPressEvent(event)
                else:
                    self.enter_pressed.emit()
            else:
                super().keyPressEvent(event)

    class ComposerPreviewThumb(QFrame):
        def __init__(self, *, path: str, attachment_kind: str, remove_callback, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setObjectName("ComposerPreviewThumb")
            self.setFixedSize(88, 88)
            self.remove_callback = remove_callback

            thumb_layout = QVBoxLayout(self)
            thumb_layout.setContentsMargins(0, 0, 0, 0)
            thumb_layout.setSpacing(0)

            self.thumb_label = QLabel()
            self.thumb_label.setAlignment(Qt.AlignCenter)
            self.thumb_label.setFixedSize(88, 88)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self.thumb_label.setPixmap(
                    pixmap.scaled(88, 88, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                )
            else:
                self.thumb_label.setText("IMG" if attachment_kind == "image" else "CAM")
                self.thumb_label.setStyleSheet("font-size: 12px; font-weight: 700;")
            thumb_layout.addWidget(self.thumb_label)

            self.delete_button = QPushButton("×", self)
            self.delete_button.setObjectName("ComposerPreviewDeleteButton")
            self.delete_button.setCursor(Qt.PointingHandCursor)
            self.delete_button.setFixedSize(24, 24)
            self.delete_button.move(self.width() - 28, 4)
            self.delete_button.hide()
            self.delete_button.clicked.connect(self.remove_callback)

        def enterEvent(self, event) -> None:
            self.delete_button.show()
            super().enterEvent(event)

        def leaveEvent(self, event) -> None:
            self.delete_button.hide()
            super().leaveEvent(event)

    class ChatBubble(QWidget):
        def __init__(self, message: ChatMessage, *, language: str, align_right: bool, parent: QWidget | None = None, window: QMainWindow = None) -> None:
            super().__init__(parent)
            self.chat_window = window
            self.effective_theme = getattr(window, "effective_theme", "dark")
            outer = QHBoxLayout(self)
            outer.setContentsMargins(0 , 8, 0, 8)
            outer.setSpacing(12)

            if align_right:
                outer.addStretch(1)

            self.bubble = QFrame()
            self.bubble.setObjectName("BubbleUser" if align_right else "BubbleAI")
            self.bubble.setMaximumWidth(680)
            self.bubble.setAttribute(Qt.WA_StyledBackground, True)

            shadow = QGraphicsDropShadowEffect(self.bubble)
            shadow.setBlurRadius(15 if align_right else 10)
            shadow.setXOffset(0)
            shadow.setYOffset(8 if align_right else 4)
            shadow.setColor(QColor(0, 0, 0, 90 if align_right else 40))
            self.bubble.setGraphicsEffect(shadow)

            self.bubble_layout = QVBoxLayout(self.bubble)
            self.bubble_layout.setContentsMargins(18, 14, 18, 14)
            self.bubble_layout.setSpacing(8) # Giữ nguyên spacing

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
                self.bubble_layout.addWidget(attachment_label) # Sửa lỗi trùng lặp
                if message.attachment_kind in {"image", "camera"}:
                    pixmap = QPixmap(message.attachment_path)
                    if not pixmap.isNull():
                        self.preview_img = QLabel()
                        self.preview_img.setPixmap(pixmap.scaled(320, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        self.preview_img.setCursor(Qt.PointingHandCursor)
                        self.preview_img.mousePressEvent = lambda e: self.show_image_full(message.attachment_path)
                        self.bubble_layout.addWidget(self.preview_img)

            self.text_label = QLabel()
            self.typing_indicator = None
            self.update_display_text(message.text or "")
            self.text_label.setWordWrap(True)
            self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.text_label.setStyleSheet("font-size: 15px;")
            self.bubble_layout.addWidget(self.text_label)
            outer.addWidget(self.bubble, 0, Qt.AlignTop)

            if not align_right:
                outer.addStretch(1)

        def showEvent(self, event):
            super().showEvent(event)
            self.anim = QPropertyAnimation(self.bubble, b"pos")
            self.anim.setDuration(500)
            curr = self.bubble.pos()
            self.anim.setStartValue(QPoint(curr.x(), curr.y() + 20))
            self.anim.setEndValue(curr)
            self.anim.setEasingCurve(QEasingCurve.OutBack)
            self.anim.start()

        def show_image_full(self, path: str):
            try:
                dialog = ImagePreviewDialog(path, self.window(), effective_theme=self.window().effective_theme)
                dialog.exec()
            except Exception as e:
                print(f"Error previewing image: {e}")

        def update_display_text(self, text: str):
            if text == "[TYPING]":
                self.text_label.hide()
                if not self.typing_indicator:
                    self.typing_indicator = TypingIndicator(QColor("#4db8ff"))
                    self.bubble_layout.addWidget(self.typing_indicator)
                self.typing_indicator.show()
                return
            
            if self.typing_indicator:
                self.typing_indicator.hide()
            self.text_label.show()

            is_error = text.startswith("API Error:") or text.startswith("Error:")
            if self.effective_theme == "light" and is_error:
                error_style = 'color: #b00020; font-weight: bold; background: rgba(176,0,32,0.08); padding: 5px; border-radius: 8px;'
            elif is_error:
                error_style = 'color: #ff5252; font-weight: bold; background: rgba(255,82,82,0.1); padding: 5px; border-radius: 8px;'
            elif self.effective_theme == "light":
                error_style = 'color: #111827; background: rgba(0,0,0,0.04); padding: 5px; border-radius: 8px;'
            else:
                error_style = ''

            if self.effective_theme == "light":
                code_bg = '#f3f4f6'
                code_color = '#111827'
                inline_code_bg = '#e5e7eb'
            else:
                code_bg = '#1e1e1e'
                code_color = '#d4d4d4'
                inline_code_bg = '#444444'

            parts = re.split(r'(```[\s\S]*?```)', text)
            formatted = ""
            for part in parts:
                if part.startswith('```'):
                    code_content = part.strip('`').strip()
                    lines = code_content.split('\n', 1)
                    lang = lines[0].strip() if len(lines) > 1 else ""
                    code = lines[1] if len(lines) > 1 else lines[0]
                    
                    if PYGMENTS_AVAILABLE:
                        try:
                            lexer = get_lexer_by_name(lang) if lang else guess_lexer(code)
                            formatter = HtmlFormatter(noclasses=True, style='monokai')
                            formatted += highlight(code, lexer, formatter)
                            continue
                        except: pass
                    formatted += f'<pre style="background: {code_bg}; color: {code_color}; padding: 10px; border-radius: 5px;"><code>{code}</code></pre>'
                else:
                    p = part.replace("\n", "<br>")
                    p = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', p)
                    p = re.sub(r'`(.*?)`', fr'<code style="background: {inline_code_bg}; padding: 2px;">\1</code>', p)
                    formatted += p
            self.text_label.setText(f"<div style='{error_style}'>{formatted}</div>")

    class ChatWindow(QMainWindow):
        def __init__(self, *, title: str, initial_camera_index: int, mode_label: str, model_label: str | None) -> None:
            super().__init__()
            self.language = "vi"
            self.theme_mode = "system"
            self.effective_theme = "system"
            self.sidebar_expanded = True
            self.is_refreshing_history = False
            self.is_recording = False
            self.typing_timer = QTimer()
            self.initial_camera_index = initial_camera_index
            self.mode_label = mode_label
            self.model_label = model_label or ""
            self.pending_image_attachments: list[tuple[str, str]] = []
            self.conversations: list[Conversation] = []
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "chat_history.db")
            self.db = ChatDatabase(db_path)
            self.db.delete_conversations_by_titles(LEGACY_SEEDED_CONVERSATION_TITLES)

            self.language = self.db.get_setting("language", "vi")
            self.theme_mode = self.db.get_setting("theme", "dark")
            
            self.active_conversation_index = 0
            self.setup_tray_icon()
            self.setup_voice_animation()
            self.setWindowTitle(title)
            self.resize(1480, 920)
            self.setAcceptDrops(True)
            self.build_ui()
            self.conversations = self.db.get_all_conversations()
            if not self.conversations:
                self.conversations = [
                    Conversation(
                        title=tr(self.language, "new_chat"),
                        subtitle=tr(self.language, "today"),
                        messages=[],
                        id=None,
                    )
                ]
            self.retranslate_ui()
            QTimer.singleShot(0, self.message_input.setFocus)

        def setup_tray_icon(self) -> None:
            self.tray_icon = QSystemTrayIcon(self)
            self.tray_icon.setIcon(themed_icon("sidebar_app.svg", self.icon_color(), 24))
            self.tray_icon.show()

        def show_tray_notification(self, message: str) -> None:
            if QSystemTrayIcon.supportsMessages() and hasattr(self, "tray_icon"):
                snippet = (message[:60] + "...") if len(message) > 60 else message
                self.tray_icon.showMessage(
                    tr(self.language, "new_message"),
                    snippet,
                    QSystemTrayIcon.Information,
                    3000
                )

        def scroll_to_bottom(self) -> None:
            if hasattr(self, "scroll_area"):
                bar = self.scroll_area.verticalScrollBar()
                QTimer.singleShot(50, lambda: bar.setValue(bar.maximum()))

        def setup_voice_animation(self) -> None:
            self.pulse_anim = QVariantAnimation(self)
            self.pulse_anim.setDuration(700)
            self.pulse_anim.setStartValue(QColor("#FF5252")) # Đỏ sáng
            self.pulse_anim.setEndValue(QColor("#7F2929"))   # Đỏ tối
            self.pulse_anim.setEasingCurve(QEasingCurve.InOutSine)
            self.pulse_anim.setLoopCount(-1)
            self.pulse_anim.valueChanged.connect(self.update_mic_style)

        def update_mic_style(self, color: QColor) -> None:
            self.micro_button.setStyleSheet(f"""
                QPushButton#RoundButton {{
                    background-color: {color.name()};
                    border: 2px solid {color.name()};
                }}
            """)

        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.accept()
            else:
                event.ignore()

        def dropEvent(self, event):
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.handle_dropped_image(path)
                elif path.lower().endswith('.txt'):
                    self.handle_dropped_text(path)

        def handle_dropped_image(self, path: str):
            self.queue_image_attachment(path, "image")

        def handle_dropped_text(self, path: str):
            content = Path(path).read_text(encoding="utf-8", errors="ignore")[:500]
            self.add_message(ChatMessage(sender="user", text=f"File: {Path(path).name}\n{content}...", attachment_path=path, attachment_kind="text"))
            self.generate_ai_response("Summarize this text.", path, "text")

        def build_ui(self) -> None:
            root_widget = QWidget()
            root_widget.setObjectName("Root")
            
            self.sidebar_shadow = QGraphicsDropShadowEffect(root_widget)
            self.sidebar_shadow.setBlurRadius(30)
            self.sidebar_shadow.setXOffset(10)
            self.sidebar_shadow.setYOffset(0)
            self.sidebar_shadow.setColor(QColor(0, 0, 0, 100))
            
            root = QHBoxLayout(root_widget)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
            self.setCentralWidget(root_widget)

            self.sidebar = QFrame()
            self.sidebar.setObjectName("Sidebar")
            self.sidebar.setMinimumWidth(320)
            self.sidebar.setMaximumWidth(360)
            self.sidebar.setGraphicsEffect(self.sidebar_shadow)
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
            sidebar_layout.addWidget(self.search_compact_button, 0, Qt.AlignLeft)

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
            self.history_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Ẩn thanh cuộn dọc
            self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Ẩn thanh cuộn ngang
            self.history_list.setContextMenuPolicy(Qt.CustomContextMenu) # Kích hoạt menu ngữ cảnh
            self.history_list.customContextMenuRequested.connect(self.show_history_context_menu) # Kết nối sự kiện chuột phải
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
            chat_layout.setContentsMargins(12, 14, 12, 16)
            chat_layout.setSpacing(12)

            top_row = QHBoxLayout()
            top_row.addStretch(1)
            self.mode_badge = QLabel()
            self.mode_badge.setObjectName("Subtle")
            top_row.addWidget(self.mode_badge)
            self.theme_button = QPushButton()
            self.theme_button.setObjectName("RoundButton")
            self.theme_button.setFixedSize(36, 36)
            self.theme_button.clicked.connect(self.cycle_theme_mode)
            top_row.addWidget(self.theme_button)
            chat_layout.addLayout(top_row)

            self.greeting_card = QFrame()
            self.greeting_card.setObjectName("GreetingCard")
            greeting_layout = QVBoxLayout(self.greeting_card)
            greeting_layout.setContentsMargins(24, 18, 24, 18)
            greeting_layout.setSpacing(8)
            self.greeting_title = QLabel()
            self.greeting_title.setObjectName("Headline")
            self.greeting_text = QLabel()
            self.greeting_text.setObjectName("Subtle")
            self.greeting_text.setStyleSheet("font-size: 16px;")
            greeting_layout.addWidget(self.greeting_title)
            self.greeting_title.setAlignment(Qt.AlignCenter)
            greeting_layout.addWidget(self.greeting_text)
            self.greeting_text.setAlignment(Qt.AlignCenter)
            chat_layout.addWidget(self.greeting_card)

            self.scroll_area = QScrollArea()
            self.scroll_area.setObjectName("MessageScroll")
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setFrameShape(QFrame.NoFrame)
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.messages_host = QWidget()
            self.messages_host.setObjectName("MessagesHost")
            self.messages_layout = QVBoxLayout(self.messages_host)
            self.messages_layout.setContentsMargins(0, 6, 0, 6)
            self.messages_layout.setSpacing(16)
            self.messages_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
            self.scroll_area.setWidget(self.messages_host)
            chat_layout.addWidget(self.scroll_area, 1)

            self.recording_panel = RecordingPanel(self.language, window=self)
            self.recording_panel.hide()
            chat_layout.addWidget(self.recording_panel, 0, Qt.AlignCenter)

            self.composer = QFrame()
            self.composer.setObjectName("Composer")
            composer_layout = QVBoxLayout(self.composer)
            composer_layout.setContentsMargins(8, 8, 8, 8)
            composer_layout.setSpacing(6)
            self.composer.setMinimumHeight(82)
            self.composer.setMaximumHeight(180)

            self.image_preview_area = QScrollArea()
            self.image_preview_area.setObjectName("ComposerPreviewScroll")
            self.image_preview_area.setWidgetResizable(True)
            self.image_preview_area.setFrameShape(QFrame.NoFrame)
            self.image_preview_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.image_preview_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.image_preview_area.setFixedHeight(92)
            self.image_preview_area.hide()

            self.image_preview_host = QWidget()
            self.image_preview_host.setObjectName("ComposerPreviewHost")
            self.image_preview_layout = QHBoxLayout(self.image_preview_host)
            self.image_preview_layout.setContentsMargins(2, 2, 2, 2)
            self.image_preview_layout.setSpacing(8)
            self.image_preview_layout.addStretch(1)
            self.image_preview_area.setWidget(self.image_preview_host)
            composer_layout.addWidget(self.image_preview_area)

            self.message_input_row = QFrame()
            self.message_input_row.setObjectName("ComposerInputRow")
            input_row_layout = QHBoxLayout(self.message_input_row)
            input_row_layout.setContentsMargins(6, 4, 6, 4)
            input_row_layout.setSpacing(6)

            self.plus_button = QPushButton("")
            self.plus_button.setFixedSize(44, 44)
            self.plus_button.clicked.connect(self.show_plus_menu)
            input_row_layout.addWidget(self.plus_button, 0, Qt.AlignVCenter)

            self.message_input = MessageInput()
            self.message_input.setObjectName("ComposerInput")
            self.message_input.setMinimumHeight(40)
            self.message_input.setMaximumHeight(88)
            self.message_input.setFrameShape(QFrame.NoFrame)
            self.message_input.setPlaceholderText(tr(self.language, "input_placeholder"))
            self.message_input.viewport().setAutoFillBackground(False)
            self.message_input.viewport().setStyleSheet("background: transparent;")
            self.message_input.enter_pressed.connect(self.send_message)
            input_row_layout.addWidget(self.message_input, 1, Qt.AlignVCenter)

            self.micro_button = QPushButton("")
            self.micro_button.setObjectName("RoundButton")
            self.micro_button.setFixedSize(44, 44)
            self.micro_button.clicked.connect(self.start_voice_input)
            input_row_layout.addWidget(self.micro_button, 0, Qt.AlignVCenter)

            self.send_button = QPushButton("↑")
            self.send_button.setObjectName("SendButton")
            self.send_button.setFixedSize(44, 44)
            self.send_button.clicked.connect(self.send_message)
            input_row_layout.addWidget(self.send_button, 0, Qt.AlignVCenter)
            composer_layout.addWidget(self.message_input_row)
            chat_layout.addWidget(self.composer)

            self.search_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
            self.search_shortcut.activated.connect(self.focus_search)

            root.addWidget(self.chat_panel, 4)

        def apply_dark_theme(self) -> None:
            self.effective_theme = "dark"
            app = QApplication.instance()
            if app:
                app.setStyleSheet(DARK_STYLESHEET)
            self.apply_theme_assets()
            self.refresh_history()

        def apply_light_theme(self) -> None:
            self.effective_theme = "light"
            app = QApplication.instance()
            if app:
                app.setStyleSheet(LIGHT_STYLESHEET)
            self.apply_theme_assets()
            self.refresh_history()

        def detect_system_theme(self) -> str:
            if platform.system() == "Windows":
                try:
                    import winreg
                    registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
                    key = winreg.OpenKey(registry, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
                    value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    return "light" if value == 1 else "dark"
                except Exception:
                    pass
            return "dark"

        def apply_theme(self) -> None:
            target = self.theme_mode
            if target == "system":
                target = self.detect_system_theme()

            self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
            self.fade_anim.setDuration(150)
            self.fade_anim.setStartValue(1.0)
            self.fade_anim.setEndValue(0.7)

            def on_fade_done():
                if target == "light":
                    self.effective_theme = "light"
                    app = QApplication.instance()
                    if app:
                        app.setStyleSheet(LIGHT_STYLESHEET)
                else:
                    self.effective_theme = "dark"
                    app = QApplication.instance()
                    if app:
                        app.setStyleSheet(DARK_STYLESHEET)

                self.db.set_setting("theme", self.theme_mode)
                self.apply_theme_assets()
                self.refresh_history()

                self.fade_in = QPropertyAnimation(self, b"windowOpacity")
                self.fade_in.setDuration(200)
                self.fade_in.setStartValue(0.7)
                self.fade_in.setEndValue(1.0)
                self.fade_in.start()

            self.fade_anim.finished.connect(on_fade_done)
            self.fade_anim.start()

        def cycle_theme_mode(self) -> None:
            order = ["system", "dark", "light"]
            current = self.theme_mode if self.theme_mode in order else "system"
            next_index = (order.index(current) + 1) % len(order)
            self.theme_mode = order[next_index]
            self.apply_theme()

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
            self.theme_button.setText("\u2600" if self.effective_theme == "dark" else "\u263E")
            self.message_input.apply_visual_style(dark_mode=self.effective_theme == "dark")

        def retranslate_ui(self) -> None:
            self.new_chat_button.setText(tr(self.language, "new_chat"))
            self.search_input.setPlaceholderText(tr(self.language, "search"))
            self.history_title.setText(tr(self.language, "history"))
            self.settings_button.setText(tr(self.language, "settings"))
            self.search_input.setPlaceholderText(tr(self.language, "search"))
            self.greeting_title.setText(tr(self.language, "greeting_title"))
            self.db.set_setting("language", self.language)
            self.greeting_text.setText(tr(self.language, "greeting_text"))
            self.message_input.setPlaceholderText(tr(self.language, "input_placeholder"))
            badge = f"{tr(self.language, 'mode_badge')}: {self.mode_label}"
            if self.model_label:
                badge = f"{badge} | {self.model_label}"
            self.mode_badge.setText(badge)
            self.update_sidebar_ui()
            self.refresh_history()
            self.apply_theme()
            self.render_messages()

        def toggle_sidebar(self) -> None:
            target_width = 88 if self.sidebar_expanded else 320
            
            self.sidebar_anim = QPropertyAnimation(self.sidebar, b"minimumWidth")
            self.sidebar_anim.setDuration(250)
            self.sidebar_anim.setStartValue(self.sidebar.width())
            self.sidebar_anim.setEndValue(target_width)
            self.sidebar_anim.setEasingCurve(QEasingCurve.InOutQuad)
            
            self.sidebar_max_anim = QPropertyAnimation(self.sidebar, b"maximumWidth")
            self.sidebar_max_anim.setDuration(250)
            self.sidebar_max_anim.setStartValue(self.sidebar.width())
            self.sidebar_max_anim.setEndValue(target_width)
            self.sidebar_max_anim.setEasingCurve(QEasingCurve.InOutQuad)

            self.sidebar_group = QParallelAnimationGroup()
            self.sidebar_group.addAnimation(self.sidebar_anim)
            self.sidebar_group.addAnimation(self.sidebar_max_anim)
            
            def on_finished():
                self.sidebar_expanded = not self.sidebar_expanded
                self.update_sidebar_ui()
            
            self.sidebar_group.finished.connect(on_finished)
            self.sidebar_group.start()

        def focus_search(self) -> None:
            if not self.sidebar_expanded:
                self.sidebar_expanded = True
                self.update_sidebar_ui()
            self.search_input.setFocus()

        def update_sidebar_ui(self) -> None:
            expanded = self.sidebar_expanded
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

        def active_conversation(self) -> Conversation:
            if not self.conversations:
                self.conversations.append(
                    Conversation(
                        title=tr(self.language, "new_chat"),
                        subtitle=tr(self.language, "today"),
                        messages=[],
                        id=None,
                    )
                )
                self.active_conversation_index = 0
            return self.conversations[self.active_conversation_index]

        def refresh_history(self) -> None:
            query = self.search_input.text().strip().lower() if hasattr(self, "search_input") else ""
            
            matching_conv_ids = set()
            if query:
                matching_conv_ids = set(self.db.search_conversations_by_message(query))

            self.is_refreshing_history = True
            self.history_list.blockSignals(True)
            self.history_list.clear()
            current_item_row = 0
            visible_row = 0
            for index, conversation in enumerate(self.conversations):
                if conversation.id is None:
                    continue
                matches_title = query in conversation.title.lower()
                matches_message = conversation.id in matching_conv_ids
                
                if query and not (matches_title or matches_message):
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

        def show_history_context_menu(self, pos: QPoint) -> None:
            item = self.history_list.itemAt(pos)
            if not item:
                return
            
            index = item.data(Qt.UserRole) # Lấy chỉ mục thực của đoạn chat
            menu = QMenu(self)
            delete_action = QAction(tr(self.language, "delete_chat"), self)
            delete_action.triggered.connect(lambda: self.delete_conversation(index))
            menu.addAction(delete_action)
            menu.exec(self.history_list.mapToGlobal(pos))

        def delete_conversation(self, index_to_delete: int) -> None:
            if not (0 <= index_to_delete < len(self.conversations)):
                return
            
            conv_id = self.conversations[index_to_delete].id
            if conv_id is not None:
                self.db.delete_conversation(conv_id)

            self.conversations.pop(index_to_delete)
            if not self.conversations:
                self.conversations.append(
                    Conversation(
                        title=tr(self.language, "new_chat"),
                        subtitle=tr(self.language, "today"),
                        messages=[],
                        id=None,
                    )
                )
                self.active_conversation_index = 0
            elif self.active_conversation_index >= index_to_delete:
                self.active_conversation_index = max(0, self.active_conversation_index - 1)
            
            self.refresh_history()
            self.render_messages()

        def clear_all_history(self) -> None:
            self.db.clear_all_conversations()
            self.conversations = []
            self.start_new_chat()
            self.refresh_history()
            self.render_messages()

        def render_messages(self) -> None:
            while self.messages_layout.count() > 1:
                item = self.messages_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            
            messages = self.active_conversation().messages
            self.greeting_card.setVisible(len(messages) == 0)
            
            for message in messages:
                bubble = ChatBubble(message, language=self.language, align_right=message.sender == "user", window=self)
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
                messages=[],
                id=None,
            )
            self.conversations.insert(0, conversation)
            self.active_conversation_index = 0
            self.refresh_history()
            self.history_list.setCurrentRow(0)
            self.render_messages()

        def show_plus_menu(self) -> None:
            menu = QMenu(self)
            strong = self.icon_color()

            image_action = QAction(themed_icon("image.svg", strong, 18), tr(self.language, "choose_image"), self)
            image_action.triggered.connect(self.pick_image)
            menu.addAction(image_action)

            text_action = QAction(themed_icon("file_text.svg", strong, 18), tr(self.language, "choose_text"), self)
            text_action.triggered.connect(self.pick_text_file)
            menu.addAction(text_action)

            camera_action = QAction(themed_icon("camera.svg", strong, 18), tr(self.language, "camera"), self)
            camera_action.triggered.connect(self.open_camera)
            menu.addAction(camera_action)

            menu.exec(self.plus_button.mapToGlobal(self.plus_button.rect().bottomLeft()))

        def add_message(self, message: ChatMessage) -> None:
            conversation = self.active_conversation()
            if conversation.id is None:
                conversation.id = self.db.create_conversation(conversation.title, conversation.subtitle)
            message.id = self.db.add_message(conversation.id, message)
            conversation.messages.append(message)
            if conversation.title == tr(self.language, "new_chat") and message.sender == "user":
                first_line = message.text.strip().splitlines()[0] if message.text.strip() else ""
                new_title = (first_line[:28] or tr(self.language, "new_chat")).strip()
                conversation.title = new_title
                self.db.update_conversation_title(conversation.id, new_title)
            conversation.subtitle = tr(self.language, "today")
            self.refresh_history()
            self.render_messages()

        def queue_image_attachment(self, path: str, attachment_kind: str = "image") -> None:
            self.pending_image_attachments.append((path, attachment_kind))
            self.refresh_image_previews()
            self.message_input.setFocus()

        def refresh_image_previews(self) -> None:
            while self.image_preview_layout.count() > 1:
                item = self.image_preview_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            has_attachments = bool(self.pending_image_attachments)
            self.image_preview_area.setVisible(has_attachments)
            self.composer.setMinimumHeight(152 if has_attachments else 82)

            if not has_attachments:
                return

            for index, (path, attachment_kind) in enumerate(self.pending_image_attachments):
                thumb_widget = ComposerPreviewThumb(
                    path=path,
                    attachment_kind=attachment_kind,
                    remove_callback=lambda _checked=False, idx=index: self.remove_pending_image_attachment(idx),
                )
                self.image_preview_layout.insertWidget(self.image_preview_layout.count() - 1, thumb_widget)

        def remove_pending_image_attachment(self, index: int) -> None:
            if not (0 <= index < len(self.pending_image_attachments)):
                return
            self.pending_image_attachments.pop(index)
            self.refresh_image_previews()
            self.message_input.setFocus()

        def clear_pending_image_previews(self) -> None:
            self.pending_image_attachments.clear()
            self.refresh_image_previews()

        def pending_attachment_prompt(self) -> str:
            if not self.pending_image_attachments:
                return ""
            if len(self.pending_image_attachments) == 1:
                _, attachment_kind = self.pending_image_attachments[0]
                return tr(self.language, "attach_camera_label") if attachment_kind == "camera" else tr(self.language, "attach_image_label")
            return f"{len(self.pending_image_attachments)} attachments"

        def send_message(self) -> None:
            text = self.message_input.toPlainText().strip()
            if not text and not self.pending_image_attachments:
                QMessageBox.information(self, tr(self.language, "info_title"), tr(self.language, "empty_send"))
                return

            for index, (path, attachment_kind) in enumerate(self.pending_image_attachments):
                attachment_text = text if index == 0 and text else (
                    tr(self.language, "attach_camera_label") if attachment_kind == "camera" else tr(self.language, "attach_image_label")
                )
                self.add_message(
                    ChatMessage(
                        sender="user",
                        text=attachment_text,
                        attachment_path=path,
                        attachment_kind=attachment_kind,
                    )
                )

            if text and not self.pending_image_attachments:
                self.add_message(ChatMessage(sender="user", text=text))

            self.message_input.clear()
            prompt = text or self.pending_attachment_prompt()
            first_attachment = self.pending_image_attachments[0] if self.pending_image_attachments else None
            self.clear_pending_image_previews()
            self.generate_ai_response(
                prompt,
                first_attachment[0] if first_attachment else None,
                first_attachment[1] if first_attachment else None,
            )

        def generate_ai_response(self, prompt: str, attach_path: str = None, attach_kind: str = None):
            self.add_message(ChatMessage(sender="ai", text=tr(self.language, "ai_unavailable")))
            self.scroll_to_bottom()

        def start_typewriter(self, full_text: str):
            self.typewriter_content = full_text
            self.typewriter_idx = 0
            self.current_ai_text = ""
            
            if self.typing_timer.isActive():
                self.typing_timer.stop()
            
            self.typewriter_tick()

        def typewriter_tick(self):
            if self.typewriter_idx < len(self.typewriter_content):
                char = self.typewriter_content[self.typewriter_idx]
                self.current_ai_text += char
                self.update_last_message(self.current_ai_text)
                self.typewriter_idx += 1
                
                delay = random.randint(10, 45)
                
                if char in ".?!":
                    delay += 400
                elif char in ",;:":
                    delay += 200
                
                QTimer.singleShot(delay, self.typewriter_tick)
                self.scroll_to_bottom()
            else:
                self.show_tray_notification(self.typewriter_content)

        def update_last_message(self, text: str):
            conv = self.active_conversation()
            if conv.messages:
                conv.messages[-1].text = text
                idx = self.messages_layout.count() - 2
                if idx >= 0:
                    item = self.messages_layout.itemAt(idx)
                    if item and item.widget():
                        last_bubble = item.widget()
                        if isinstance(last_bubble, ChatBubble):
                            last_bubble.update_display_text(text)
                            if text.startswith("Error:") or text.startswith("API Error:"):
                                self.shake_bubble(last_bubble)

        def shake_bubble(self, bubble: ChatBubble):
            orig_pos = bubble.pos()
            shake = QPropertyAnimation(bubble, b"pos", self)
            shake.setDuration(300)
            shake.setStartValue(orig_pos)
            shake.setKeyValueAt(0.15, orig_pos + QPoint(-8, 0))
            shake.setKeyValueAt(0.3, orig_pos + QPoint(8, 0))
            shake.setKeyValueAt(0.45, orig_pos + QPoint(-6, 0))
            shake.setKeyValueAt(0.6, orig_pos + QPoint(6, 0))
            shake.setKeyValueAt(0.75, orig_pos + QPoint(-3, 0))
            shake.setKeyValueAt(0.9, orig_pos + QPoint(3, 0))
            shake.setEndValue(orig_pos)
            shake.setEasingCurve(QEasingCurve.OutBounce)
            shake.start(QPropertyAnimation.DeleteWhenStopped)

        def start_voice_input(self) -> None:
            if not VOICE_AI_AVAILABLE:
                return

            if self.is_recording:
                self.worker.stop()
                return

            self.is_recording = True
            self.message_input.setPlaceholderText(tr(self.language, "loading_model"))
            self.message_input.setEnabled(False)
            self.recording_panel.show()
            self.pulse_anim.start()

            self.worker = VoiceWorker(self.language)
            self.worker.result_ready.connect(self.on_voice_success)
            self.worker.intensity_changed.connect(self.recording_panel.waveform.set_intensity)
            self.worker.error.connect(self.on_voice_error)
            self.worker.finished.connect(self.on_voice_complete)
            self.worker.start()
            self.message_input.setPlaceholderText(tr(self.language, "recording_status"))

        def on_voice_success(self, text: str):
            if text:
                existing_text = self.message_input.toPlainText().strip()
                combined_text = f"{existing_text} {text}".strip() if existing_text else text
                self.message_input.setPlainText(combined_text)

        def on_voice_error(self):
            QMessageBox.warning(self, tr(self.language, "info_title"), tr(self.language, "voice_error"))

        def on_voice_complete(self):
            self.is_recording = False
            self.pulse_anim.stop()
            self.recording_panel.hide()
            self.micro_button.setStyleSheet("")
            self.micro_button.setIcon(themed_icon("mic.svg", self.icon_color(), 18))
            self.message_input.setPlaceholderText(tr(self.language, "input_placeholder"))
            self.message_input.setEnabled(True)
            self.message_input.setFocus()

        def build_ai_reply(self, *, text: str, source: str) -> str:
            if source == "image":
                return tr(self.language, "ai_reply_image")
            if source == "camera":
                return tr(self.language, "ai_reply_camera")
            if source == "text_file":
                return tr(self.language, "ai_reply_text")
            return f"{tr(self.language, 'ai_reply_text')} {text[:120]}".strip()

        def _format_error(self, err: str) -> str:
            if "does not support image input" in err or "Cannot read" in err:
                return tr(self.language, "image_model_error")
            return f"API Error: {err}"

        def pick_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                tr(self.language, "choose_image"),
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
            )
            if not path:
                return
            self.queue_image_attachment(path, "image")

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
            self.queue_image_attachment(path, "camera")

        def open_settings(self) -> None:
            SettingsDialog(parent_window=self).exec()

    app = QApplication.instance() or QApplication(sys.argv)
    window = ChatWindow(title=window_title, initial_camera_index=camera_index, mode_label=app_mode, model_label=selected_model)
    window.show()
    return app.exec()
