import sys
import os
import pickle
import numpy as np
import pandas as pd
import random
import time
import io
import zipfile
import hashlib
import uuid
import json
from datetime import datetime, timedelta, UTC
import re

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import base64
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import keyring

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QStackedWidget, QSpacerItem, QSizePolicy,
                             QMessageBox, QDialog, QTextEdit,
                             QDialogButtonBox, QFrame, QGraphicsDropShadowEffect,
                             QCheckBox)
from PyQt6.QtGui import QFont, QColor, QMouseEvent, QPainter, QPen, QBrush
from PyQt6.QtCore import Qt, QEvent, pyqtSignal, QRect

# Add Matplotlib imports
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# STYLESHEET
APP_STYLESHEET = """
QWidget {
    font-family: 'Segoe UI', sans-serif;
}
QMainWindow {
    background-color: #0F0F0F;
}
#Card {
    background-color: rgb(24, 24, 24);
    border-radius: 8px;
}
QLabel {
    background-color: transparent;
    color: #A0A0A0;
    font-size: 14px;
}
#TitleLabel {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: bold;
    font-family: 'Courier New', 'Consolas', 'Lucida Console', monospace;
}
#DashboardTitle {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: bold;
    padding-bottom: 4px;
}
#StatValue {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: bold;
}
#StatLabel {
    color: #A0A0A0;
    font-size: 11px;
}
#FieldLabel {
    color: #FFFFFF;
    font-size: 13px;
    font-weight: bold;
    padding-bottom: 2px;
}
#QuoteLabel {
    color: #D0D0D0;
    font-size: 13px;
    line-height: 150%;
    font-family: 'Courier New', 'Consolas', 'Lucida Console', monospace;
    font-weight: bold;
}
QLineEdit, QTextEdit {
    background-color: #333333;
    border: 1px solid #333333;
    border-radius: 8px;
    color: #FFFFFF;
    padding: 10px;
    font-size: 14px;
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #5865F2;
}
#NextButton {
    background-color: #5865F2;
    color: #FFFFFF;
    border-radius: 8px;
    font-size: 14px;
    font-weight: bold;
    padding: 10px;
    border: none;
}
#NextButton:hover {
    background-color: #6974f3;
}
#NextButton:disabled {
    background-color: #404780;
    color: #9093b0;
}
#LinkButton {
    color: #00A8FC;
    background-color: transparent;
    text-align: left;
    border: none;
    font-size: 13px;
}
#LinkButton:hover {
    color: #38bcfc;
}
#AdminLinkButton {
    color: #00A8FC;
    background-color: transparent;
    border: none;
    font-size: 13px;
}
#AdminLinkButton:hover {
    color: #38bcfc;
}
QCheckBox {
    color: #A0A0A0;
    font-size: 13px;
    background-color: transparent;
}
QMessageBox {
    background-color: #181818;
}
QMessageBox QLabel {
    color: #E0E0E0;
    font-size: 14px;
}
QMessageBox QPushButton {
    background-color: #4A4A4A;
    color: #FFFFFF;
    border: none;
    padding: 8px 24px;
    border-radius: 8px;
    min-width: 80px;
}
QMessageBox QPushButton:hover {
    background-color: #5A5A5A;
}
"""

# CONSTANTS AND PATHS 
MODEL_PATH = './model/esn_svm.pkl'
TEMPLATE_DIR = './data/app_data/'
ENROLL_QUOTES_PATH = './data/raw/enroll_quotes.csv'
VERIFY_QUOTES_PATH = './data/raw/verify_quotes.csv'
DICTIONARY_PATH = './data/raw/dictionary.txt'
LOG_FILE_PATH = os.path.join(TEMPLATE_DIR, 'secure_audit.log')
ADMIN_CONFIG_PATH = os.path.join(TEMPLATE_DIR, 'admin.cfg')

# Application Info
APP_NAME = "KeystrokeDynamics"
APP_VERSION = "9.0.4" # Version incremented for gibberish check feature

# Keystroke Config
NUM_ENROLL_SAMPLES = 3

# Accuracy Config
MIN_CHAR_ACCURACY = 93.0
MIN_WORD_ACCURACY = 80.0

# Rate Limiting Config
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_PERIOD_SECONDS = 900 # 15 minutes

# Security Thresholds
MIN_SECURE_SVM_THRESHOLD = 0.70
MIN_SECURE_COSINE_THRESHOLD = 0.90 # Fallback only

# Multi-layer defense thresholds (Optimized from simulation)
CONFIDENCE_FLOOR = 0.67
SUSPICIOUS_SCORE_THRESHOLD = 0.75

# Adaptive Threshold Config
MAX_WINDOW_SIZE = 10
ANCHOR_WEIGHT = 0.7
WINDOW_WEIGHT = 0.3
DRIFT_SESSIONS_FOR_REANCHOR = 5
MIN_SAMPLES_FOR_DYNAMIC_THRESH = 4
TIGHT_CONSISTENCY_STD_DEV = 0.18 # Fallback for old profiles
LOOSE_CONSISTENCY_STD_DEV = 0.30 # Fallback for old profiles

# Proactive Health & Security Config
PROACTIVE_HEALTH_THRESHOLD = 40
PROACTIVE_MIN_SAMPLES = 5
CONSISTENCY_WEIGHT = 0.5
PERFORMANCE_WEIGHT = 0.5
PROACTIVE_SNOOZE_SESSIONS = 15
CONSECUTIVE_ANOMALY_LIMIT = 3
MAX_QUARANTINE_SIZE = 20

# Gibberish Check Config
GIBBERISH_VALIDITY_THRESHOLD = 60.0 # <-- NEW

# Security & Logging Config
KEYRING_SERVICE_NAME = 'KeystrokeDynamicsApp'
SECRET_DERIVATION_SALT = b'\x8a\x0b\x2d\x1f\x9c\x0e\x4a\xd3\xbf\x7e\x6d\x5c\x89\xab\xcd\xef'
KDF_ITERATIONS = 100000
LOG_RETENTION_DAYS = 90
HIGHLIGHT_STYLE = "background-color: #333333; color: #D0D0D0; border-radius: 3px; padding: 0px 2px;"


# GLOBAL DATA LOADING
MODEL = None
ENROLL_QUOTES = []
VERIFY_QUOTES = []
DICTIONARY_WORDS = set() # <-- NEW

def load_model():
    global MODEL
    try:
        with open(MODEL_PATH, 'rb') as f:
            MODEL = pickle.load(f)
        return True
    except FileNotFoundError:
        QMessageBox.critical(None, "Startup Error", f"FATAL ERROR: Model file not found at {MODEL_PATH}")
        return False

def load_quotes_from_csv(path, quote_list, is_verify_list=False):
    try:
        df = pd.read_csv(path, header=None, names=['Quote'], on_bad_lines='skip')
        if 'Quote' not in df.columns:
            raise ValueError("CSV must contain a 'Quote' column.")
        quotes_data = df['Quote'].dropna().tolist()
        if is_verify_list:
            default_quote_text = "What we plant in the soil of contemplation, we shall reap in the harvest of action."
            if not any(quote_text.startswith(default_quote_text) for quote_text in quotes_data):
                quotes_data.insert(0, default_quote_text)
        quote_list.extend(quotes_data)
        return True
    except FileNotFoundError:
        QMessageBox.critical(None, "Startup Error", f"Quote file not found: {path}")
        return False
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", f"Error reading quote file {path}: {e}")
        return False

def load_dictionary(path, word_set):
    """Loads words from the dictionary file into a set."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Use a generator expression for memory efficiency and strip newlines
            words = (line.strip().lower() for line in f)
            word_set.update(words)
        if not word_set:
            QMessageBox.critical(None, "Startup Error", f"Dictionary file is empty or could not be read: {path}")
            return False
        return True
    except FileNotFoundError:
        QMessageBox.critical(None, "Startup Error", f"Dictionary file not found: {path}")
        return False
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", f"Error reading dictionary file {path}: {e}")
        return False


# DIALOGS AND CUSTOM WIDGETS
class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class ReAnchorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Re-Anchoring Authorization")
        self.setStyleSheet("""
            QDialog { background-color: #161b2; }
            QLabel { color: #c9d1d9; font-size: 13px; }
            QLineEdit { background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 14px; }
            QLineEdit:focus { border: 1px solid #58a6ff; }
            QPushButton { min-width: 90px; background-color: #238636; color: #fff; border: 1px solid #2ea043; padding: 8px; border-radius: 6px; }
            """)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Your typing pattern has drifted. Please provide your password to update the profile:"))
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(QLabel("Password:"))
        layout.addWidget(self.password_entry)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_password(self):
        return self.password_entry.text()

class MandatoryReAnchorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Security Alert: Action Required")
        self.setStyleSheet("""
            QDialog { background-color: #2d1313; border: 1px solid #DA3633; }
            QLabel { color: #f0c6c6; font-size: 13px; }
            QLineEdit { background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 14px; }
            QLineEdit:focus { border: 1px solid #DA3633; }
            QPushButton { min-width: 90px; background-color: #DA3633; color: #fff; border: 1px solid #f85149; padding: 8px; border-radius: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #b22a27; }
            """)
        layout = QVBoxLayout(self)
        warning_message = ("<b>Warning:</b> Multiple consecutive unusual login attempts have been detected on your account. "
                           "To secure your profile, please verify with your password to create a new trusted baseline.")
        layout.addWidget(QLabel(warning_message, wordWrap=True))
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(QLabel("Password:"))
        layout.addWidget(self.password_entry)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_password(self):
        return self.password_entry.text()

class StepUpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Additional Verification Required")
        self.setStyleSheet("""
            QDialog { background-color: #181818; }
            QLabel { color: #c9d1d9; font-size: 13px; }
            QLineEdit { background-color: #333333; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 14px; }
            QLineEdit:focus { border: 1px solid #5865F2; }
            QPushButton {
                min-width: 90px;
                background-color: #5865F2;
                color: #fff;
                border: none;
                padding: 8px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #6974f3;
            }
            """)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Your typing pattern was unusual. Please provide your password to continue:"))
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(QLabel("Password:"))
        layout.addWidget(self.password_entry)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_password(self):
        return self.password_entry.text()

class ProactiveReAnchorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profile Maintenance Recommended")
        self.setStyleSheet("""
            QDialog { background-color: #181818; }
            QLabel { color: #c9d1d9; font-size: 13px; }
            QPushButton {
                min-width: 120px;
                background-color: #5865F2;
                color: #fff; border: none;
                padding: 8px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #6974f3; }
            #RemindButton { background-color: #4A4A4A; }
            #RemindButton:hover { background-color: #5A5A5A; }
            """)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        message = ("Your typing has been less consistent recently. To ensure smooth access "
                   "in the future, we recommend updating your profile's anchor point now.")
        layout.addWidget(QLabel(message))

        buttons = QDialogButtonBox()
        update_button = QPushButton("Update Now")
        remind_button = QPushButton("Remind Me Later")
        remind_button.setObjectName("RemindButton")

        buttons.addButton(update_button, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(remind_button, QDialogButtonBox.ButtonRole.RejectRole)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

class LogViewerDialog(QDialog):
    def __init__(self, logs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Secure Audit Log Viewer")
        self.setGeometry(150, 150, 800, 600)
        self.setStyleSheet("""
            QDialog { background-color: #0F0F0F; }
            QTextEdit { background-color: #181818; color: #D0D0D0; font-family: 'Consolas', 'Courier New', monospace; border: 1px solid #30363d; }
            """)
        layout = QVBoxLayout(self)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)
        sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''), reverse=True)
        self.log_display.setText("\n".join(json.dumps(log, indent=2) for log in sorted_logs) or "No log entries found.")

class HealthGaugeWidget(QWidget):
    """A widget to display a health score as a circular gauge."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self.setMinimumSize(120, 120)

    def setValue(self, value):
        self._value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        centered_rect = QRect(int((self.width() - side) / 2), int((self.height() - side) / 2), side, side)
        rect = centered_rect.adjusted(8, 8, -8, -8)
        
        start_angle = 225
        total_span = -270

        pen = QPen(QColor("#333333"), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, start_angle * 16, total_span * 16)

        if self._value > 70:
            color = QColor("#238636") # Green
        elif self._value > 40:
            color = QColor("#D29922") # Yellow
        else:
            color = QColor("#DA3633") # Red

        pen.setColor(color)
        pen.setWidth(11)
        painter.setPen(pen)
        
        span_angle = int((self._value / 100.0) * total_span)
        painter.drawArc(rect, start_angle * 16, span_angle * 16)
        
        font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        painter.setFont(font)
        pen.setColor(QColor("#FFFFFF"))
        painter.setPen(pen)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{int(self._value)}%")

class MplChartWidget(QWidget):
    """A widget to display a Matplotlib chart."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 3), dpi=100))
        self.ax = self.canvas.figure.subplots()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setup_chart()

    def setup_chart(self):
        self.canvas.figure.patch.set_facecolor('#181818')
        self.ax.set_facecolor('#181818')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#4A4A4A')
        self.ax.spines['bottom'].set_color('#4A4A4A')
        self.ax.tick_params(axis='x', colors='#A0A0A0', labelsize=8)
        self.ax.tick_params(axis='y', colors='#A0A0A0', labelsize=8)
        self.ax.set_ylim(0, 1.2)

    def update_chart(self, scores):
        self.ax.clear()
        self.setup_chart()
        if not scores:
            self.ax.text(0.5, 0.5, 'No score data available.', ha='center', va='center', color='#A0A0A0', fontsize=10)
        else:
            self.ax.plot(scores, marker='o', color='#5865F2', markersize=5, markerfacecolor='#FFFFFF')
            for i, score in enumerate(scores):
                self.ax.text(i, score + 0.05, f'{score:.2f}', ha='center', va='bottom', color='#D0D0D0', fontsize=8)
        self.canvas.figure.tight_layout(pad=0.5)
        self.canvas.draw()


# SECURE LOGGER IMPLEMENTATION
class SecureLogger:
    def __init__(self, log_path, retention_days):
        self.log_path, self.retention_period = log_path, timedelta(days=retention_days)
        self.fernet = Fernet(self._get_log_encryption_key())
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path): open(self.log_path, 'w').close()

    def _get_log_encryption_key(self):
        secret = keyring.get_password(KEYRING_SERVICE_NAME, 'log_encryption_key')
        if not secret:
            secret = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
            keyring.set_password(KEYRING_SERVICE_NAME, 'log_encryption_key', secret)
        kdf = PBKDF2HMAC(hashes.SHA256(), 32, SECRET_DERIVATION_SALT, KDF_ITERATIONS, default_backend())
        return base64.urlsafe_b64encode(kdf.derive(secret.encode()))

    def log_event(self, event_data):
        log_entry = event_data
        try:
            with open(self.log_path, 'ab') as f:
                f.write(self.fernet.encrypt(json.dumps(log_entry).encode('utf-8')) + b'\n')
        except Exception as e:
            print(f"Error writing to log: {e}")

    def read_logs(self):
        if not os.path.exists(self.log_path): return []
        logs = []
        with open(self.log_path, 'rb') as f:
            for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(self.fernet.decrypt(line.strip())))
                    except (InvalidToken, Exception) as e:
                        print(f"Error reading log line: {e}")
        return logs

    def purge_old_logs(self):
        all_logs, now, valid_logs = self.read_logs(), datetime.now(UTC), []
        for log in all_logs:
            log_time_str = log['timestamp'].replace('Z', '+00:00')
            if now - datetime.fromisoformat(log_time_str) < self.retention_period:
                valid_logs.append(log)
        if len(valid_logs) < len(all_logs):
            with open(self.log_path, 'wb') as f:
                for log in valid_logs:
                    f.write(self.fernet.encrypt(json.dumps(log).encode('utf-8')) + b'\n')


# MAIN APPLICATION
class KeystrokeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_session_id = None
        self.current_session_token = None # For anti-replay
        self.dashboard_data = {}
        self.logger = SecureLogger(LOG_FILE_PATH, LOG_RETENTION_DAYS)
        self.logger.purge_old_logs()
        self.setWindowTitle("Keystroke Dynamics")

        self.resize(440, 580)

        screen = QApplication.primaryScreen()
        available_size = screen.availableGeometry()
        self.move(available_size.center() - self.rect().center())

        self.press_times, self.completed_events, self.enrollment_step, self.enrollment_samples = {}, [], 0, []
        self.enroll_username, self.enroll_password = "", ""

        self.last_auth_was_success = False
        self.is_re_enrolling = False
        
        self.failed_attempts = {} # For rate limiting

        self.current_enroll_quote_data = random.choice(ENROLL_QUOTES) if ENROLL_QUOTES else "Please check enroll_quotes.csv"
        self.current_verify_quote_data = random.choice(VERIFY_QUOTES) if VERIFY_QUOTES else "Please check verify_quotes.csv"

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.login_widget = self.create_login_widget()
        self.enroll_widget = self.create_enroll_widget()
        self.admin_widget = self.create_admin_widget()

        self.stacked_widget.addWidget(self.login_widget)
        self.stacked_widget.addWidget(self.enroll_widget)
        self.stacked_widget.addWidget(self.admin_widget)

        self.dashboard_widget = self.create_dashboard_widget()
        self.stacked_widget.addWidget(self.dashboard_widget)

        self.go_to_login_page()

    def create_styled_card(self, add_shadow=True):
        card = QFrame()
        card.setObjectName("Card")
        card.setFixedSize(400, 520)
        if add_shadow:
            shadow = QGraphicsDropShadowEffect(blurRadius=1, xOffset=0, yOffset=0)
            shadow.setColor(QColor(128, 128, 128, 255))
            card.setGraphicsEffect(shadow)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(8)
        return card, layout

    def create_input_group(self, label_text, is_password=False):
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)
        label = QLabel(label_text, objectName="FieldLabel")
        line_edit = QLineEdit()
        line_edit.setFixedSize(350, 40)
        if is_password:
            line_edit.setEchoMode(QLineEdit.EchoMode.Password)
        group_layout.addWidget(label)
        group_layout.addWidget(line_edit)
        return group_layout, line_edit

    def create_login_widget(self):
        widget = QWidget()
        outer_layout = QHBoxLayout(widget)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card, layout = self.create_styled_card()
        self.login_input_widget = QWidget()
        login_input_layout = QVBoxLayout(self.login_input_widget)
        login_input_layout.setContentsMargins(0,0,0,0)
        login_input_layout.setSpacing(8)
        self.status_widget = QWidget()
        status_layout = QVBoxLayout(self.status_widget)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.setContentsMargins(0,0,0,0)
        status_layout.setSpacing(10)
        header_layout = QHBoxLayout()
        icon_label = QLabel("⌨️")
        icon_label.setFont(QFont("Segoe UI Symbol", 16))
        header_layout.addWidget(icon_label)
        header_layout.addWidget(QLabel("Keystroke Dynamics", objectName="TitleLabel"))
        header_layout.addStretch()
        login_input_layout.addLayout(header_layout)
        login_input_layout.addWidget(QLabel("Recognize users through their typing style"))
        login_input_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        username_layout, self.login_username_entry = self.create_input_group("Username (required)")
        login_input_layout.addLayout(username_layout)
        login_input_layout.addSpacerItem(QSpacerItem(20, 5))
        login_input_layout.addWidget(QLabel("Verify", objectName="FieldLabel"))
        self.login_quote_label = QLabel(wordWrap=True, objectName="QuoteLabel")
        login_input_layout.addWidget(self.login_quote_label)
        self.free_type_checkbox = QCheckBox("Type freely")
        self.free_type_checkbox.stateChanged.connect(self.toggle_free_type_mode)
        login_input_layout.addWidget(self.free_type_checkbox)
        self.login_typing_entry = QTextEdit()
        self.login_typing_entry.setPlaceholderText("Type the text shown above...")
        self.login_typing_entry.setFixedSize(350, 120)
        self.login_typing_entry.installEventFilter(self)
        login_input_layout.addWidget(self.login_typing_entry)
        login_input_layout.addStretch(1)
        self.verify_button = QPushButton("Verify", objectName="NextButton", cursor=Qt.CursorShape.PointingHandCursor)
        self.verify_button.setFixedSize(350, 40)
        self.verify_button.clicked.connect(self.verify_submit)
        self.verify_button.setEnabled(False)
        login_input_layout.addWidget(self.verify_button)
        self.login_typing_entry.textChanged.connect(self._on_login_text_changed)
        links_layout = QHBoxLayout()
        links_layout.addWidget(QLabel("New user?"))
        create_profile_link = QPushButton("Create a profile", objectName="LinkButton", cursor=Qt.CursorShape.PointingHandCursor)
        create_profile_link.clicked.connect(self.go_to_enroll_page)
        links_layout.addWidget(create_profile_link)
        links_layout.addStretch()
        admin_link = QPushButton("Admin", objectName="AdminLinkButton", cursor=Qt.CursorShape.PointingHandCursor)
        admin_link.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        links_layout.addWidget(admin_link)
        login_input_layout.addLayout(links_layout)

        self.status_icon_label = QLabel()
        self.status_icon_label.setFixedSize(100, 100)
        self.status_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_text_label = QLabel()
        self.status_text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_text_label.setObjectName("TitleLabel")
        self.status_text_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        
        self.retry_button = QPushButton("Retry?", objectName="AdminLinkButton", cursor=Qt.CursorShape.PointingHandCursor)
        self.retry_button.clicked.connect(self.go_to_login_page)

        self.dashboard_button = QPushButton("Go to Dashboard →", objectName="LinkButton", cursor=Qt.CursorShape.PointingHandCursor)
        self.dashboard_button.clicked.connect(self.show_dashboard_with_data)
        self.dashboard_button.hide() 

        status_buttons_layout = QHBoxLayout()
        status_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_buttons_layout.setSpacing(20)
        status_buttons_layout.addWidget(self.retry_button)
        status_buttons_layout.addWidget(self.dashboard_button)

        status_layout.addStretch()
        status_layout.addWidget(self.status_icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_text_label, alignment=Qt.AlignmentFlag.AlignCenter)
        status_layout.addLayout(status_buttons_layout)
        status_layout.addStretch()
        
        layout.addWidget(self.login_input_widget)
        layout.addWidget(self.status_widget)
        self.status_widget.hide()
        outer_layout.addWidget(card)
        return widget

    def create_enroll_widget(self):
        widget = QWidget()
        outer_layout = QHBoxLayout(widget)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card, layout = self.create_styled_card()
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("⌨️", font=QFont("Segoe UI Symbol", 16)))
        header_layout.addWidget(QLabel("Create a Profile", objectName="TitleLabel"))
        header_layout.addStretch()
        layout.addLayout(header_layout)
        layout.addSpacerItem(QSpacerItem(20, 10))
        username_layout, self.enroll_username_entry = self.create_input_group("Username")
        password_layout, self.enroll_password_entry = self.create_input_group("Password", is_password=True)
        layout.addLayout(username_layout)
        layout.addLayout(password_layout)
        layout.addSpacerItem(QSpacerItem(20, 5))
        self.enroll_prompt_label = QLabel(objectName="FieldLabel")
        layout.addWidget(self.enroll_prompt_label)
        self.enroll_quote_label = QLabel("Quote will appear here.", wordWrap=True, objectName="QuoteLabel")
        layout.addWidget(self.enroll_quote_label)
        self.enroll_typing_entry = QTextEdit()
        self.enroll_typing_entry.setPlaceholderText("Type the enrollment text here...")
        
        self.enroll_typing_entry.setFixedSize(350, 80)

        self.enroll_typing_entry.installEventFilter(self)
        layout.addWidget(self.enroll_typing_entry)
        
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        footer_layout = QVBoxLayout()
        footer_layout.setSpacing(8)

        self.re_enroll_checkbox = QCheckBox("Re-enroll (overwrite existing profile)")
        footer_layout.addWidget(self.re_enroll_checkbox)

        self.enroll_submit_button = QPushButton("Next Sample", objectName="NextButton", cursor=Qt.CursorShape.PointingHandCursor)
        self.enroll_submit_button.setFixedSize(350, 40)
        self.enroll_submit_button.clicked.connect(self.enroll_submit)
        self.enroll_submit_button.setEnabled(False)
        footer_layout.addWidget(self.enroll_submit_button)
        self.enroll_typing_entry.textChanged.connect(self._on_enroll_text_changed)
        
        back_link = QPushButton("← Back to login", objectName="AdminLinkButton", cursor=Qt.CursorShape.PointingHandCursor)
        back_link.clicked.connect(self.go_to_login_page)
        footer_layout.addWidget(back_link, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addLayout(footer_layout)
        outer_layout.addWidget(card)
        return widget

    def create_admin_widget(self):
        widget = QWidget()
        outer_layout = QHBoxLayout(widget)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card, layout = self.create_styled_card()
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("⌨️", font=QFont("Segoe UI Symbol", 16)))
        header_layout.addWidget(QLabel("Administrative Access", objectName="TitleLabel"))
        header_layout.addStretch()
        layout.addLayout(header_layout)
        layout.addSpacerItem(QSpacerItem(20, 15))
        username_layout, self.admin_username_entry = self.create_input_group("Username")
        password_layout, self.admin_password_entry = self.create_input_group("Password", is_password=True)
        layout.addLayout(username_layout)
        layout.addLayout(password_layout)
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        login_button = QPushButton("View Logs", objectName="NextButton", cursor=Qt.CursorShape.PointingHandCursor)
        login_button.setFixedSize(350, 40)
        login_button.clicked.connect(self.admin_login)
        layout.addWidget(login_button)
        back_link = QPushButton("← Back to login", objectName="AdminLinkButton", cursor=Qt.CursorShape.PointingHandCursor)
        back_link.clicked.connect(self.go_to_login_page)
        layout.addWidget(back_link, alignment=Qt.AlignmentFlag.AlignCenter)
        outer_layout.addWidget(card)
        return widget

    def create_dashboard_widget(self):
        widget = QWidget()
        outer_layout = QHBoxLayout(widget)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card, layout = self.create_styled_card()
    
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("⌨️", font=QFont("Segoe UI Symbol", 14)))
        header_layout.addWidget(QLabel("Dashboard", objectName="TitleLabel"))
        header_layout.addStretch()
        layout.addLayout(header_layout)
        layout.addSpacing(5)

        # Profile Health Section
        health_section_layout = QHBoxLayout()
        
        health_info_layout = QVBoxLayout()
        health_info_layout.addWidget(QLabel("Profile Health", objectName="DashboardTitle"))
        self.consistency_label = QLabel("ㅤConsistency: -")
        self.performance_label = QLabel("ㅤPerformance: -")
        health_info_layout.addWidget(self.consistency_label)
        health_info_layout.addWidget(self.performance_label)
        health_info_layout.addStretch() 
        
        health_section_layout.addLayout(health_info_layout)
        
        self.health_gauge = HealthGaugeWidget()
        self.health_gauge.setFixedSize(90, 90)
        health_section_layout.addWidget(self.health_gauge, 0, Qt.AlignmentFlag.AlignVCenter)
        
        layout.addLayout(health_section_layout)
        layout.addSpacing(20)
        
        # Score Trend Section
        layout.addWidget(QLabel("Score Trend", objectName="DashboardTitle"))
        self.score_chart = MplChartWidget()
        self.score_chart.setMinimumHeight(130)
        self.score_chart.setMaximumHeight(130)
        layout.addWidget(self.score_chart)
        layout.addSpacing(20)
        
        # Profile Stats Section
        layout.addWidget(QLabel("Profile Stats", objectName="DashboardTitle"))
        stats_layout = QHBoxLayout()
        def create_stat_box(label):
            box = QVBoxLayout()
            box.setSpacing(2)
            value_label = QLabel("0")
            value_label.setObjectName("StatValue")
            label_label = QLabel(label)
            label_label.setObjectName("StatLabel")
            box.addWidget(value_label, alignment=Qt.AlignmentFlag.AlignCenter)
            box.addWidget(label_label, alignment=Qt.AlignmentFlag.AlignCenter)
            return box, value_label

        logins_box, self.logins_value = create_stat_box("Total Logins")
        drift_box, self.drift_value = create_stat_box("Drift Status")
        anomalies_box, self.anomalies_value = create_stat_box("Anomalies")

        stats_layout.addLayout(logins_box)
        stats_layout.addStretch()
        stats_layout.addLayout(drift_box)
        stats_layout.addStretch()
        stats_layout.addLayout(anomalies_box)
        layout.addLayout(stats_layout)

        # Footer
        layout.addStretch(1)
        back_link = QPushButton("← Logout", objectName="AdminLinkButton", cursor=Qt.CursorShape.PointingHandCursor)
        back_link.clicked.connect(self.go_to_login_page)
        layout.addWidget(back_link, alignment=Qt.AlignmentFlag.AlignCenter)
    
        outer_layout.addWidget(card)
        return widget

    def show_message_box(self, title, text, icon=QMessageBox.Icon.Information, detailed_text=""):
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setText(text)
        msg_box.setWindowTitle(title)
        if detailed_text:
            msg_box.setDetailedText(detailed_text)
        msg_box.exec()

    def show_success_screen(self, message="Authenticated"):
        self.status_icon_label.setText("✓")
        self.status_icon_label.setStyleSheet("background-color: #238636; color: white; font-size: 60px; font-weight: bold; border-radius: 50px;")
        self.status_text_label.setText(message)

    def show_rejection_screen(self, message="Rejected"):
        self.status_icon_label.setText("✕")
        self.status_icon_label.setStyleSheet("background-color: #DA3633; color: white; font-size: 60px; font-weight: bold; border-radius: 50px;")
        self.status_text_label.setText(message)

    def re_anchor_prompt(self, username, metadata):
        dialog = ReAnchorDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            password = dialog.get_password()
            if self.verify_password(metadata['password_hash'], metadata['salt'], password):
                return True
            else:
                self.show_message_box("Update Failed", "Incorrect password. Profile not updated.", QMessageBox.Icon.Warning)
                return False
        return False

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress and event.key() not in [Qt.Key.Key_Tab, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Control]:
            if event.key() not in self.press_times:
                self.press_times[event.key()] = time.time()
        elif event.type() == QEvent.Type.KeyRelease and event.key() in self.press_times:
            self.completed_events.append((event.text(), self.press_times.pop(event.key()), time.time()))
        return super().eventFilter(source, event)

    def _get_user_metadata(self, username):
        dat_path = os.path.join(TEMPLATE_DIR, f"{username}.dat")
        if not os.path.exists(dat_path): return None
        try:
            with open(dat_path, 'rb') as f: encrypted_data = f.read()
            decrypted_data = Fernet(self.get_encryption_key(username)).decrypt(encrypted_data)
            with zipfile.ZipFile(io.BytesIO(decrypted_data), 'r') as zf:
                return json.loads(zf.read('metadata.json'))
        except (InvalidToken, zipfile.BadZipFile, KeyError, Exception):
            return None

    def _get_typing_pattern(self, live_timings, stored_template):
        if stored_template is None or stored_template.size == 0 or not live_timings:
            return "normal"
        live_mean_speed = np.mean([item[3] for item in live_timings])
        stored_mean_speed = stored_template[3]
        if stored_mean_speed == 0: return "normal"
        ratio = live_mean_speed / stored_mean_speed
        if ratio < 0.75: return "much_faster"
        if ratio < 0.90: return "slightly_faster"
        if ratio > 1.25: return "much_slower"
        if ratio > 1.10: return "slightly_slower"
        return "normal"

    def _calculate_levenshtein_distance(self, s1, s2):
        s1, s2 = s1.lower(), s2.lower()
        m, n = len(s1), len(s2)
        if m < n: s1, s2 = s2, s1; m, n = n, m
        previous_row = list(range(n + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions, deletions = previous_row[j + 1] + 1, current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[n]

    def _validate_text_accuracy(self, typed_text, quote_text):
        max_len = max(len(typed_text), len(quote_text))
        if max_len == 0: return False, "No text was typed."
        char_accuracy = (1 - self._calculate_levenshtein_distance(typed_text, quote_text) / max_len) * 100
        if char_accuracy < MIN_CHAR_ACCURACY: return False, "Too many typos, please re-type."
        quote_words, typed_words = set(quote_text.lower().split()), set(typed_text.lower().split())
        if not quote_words: return True, ""
        word_accuracy = (len(quote_words.intersection(typed_words)) / len(quote_words)) * 100
        if word_accuracy < MIN_WORD_ACCURACY: return False, "Too many typos, please re-type."
        return True, ""

    def go_to_enroll_page(self):
        self.current_session_id = f"sess-enroll-{uuid.uuid4().hex[:12]}"
        self.reset_keystroke_data()
        self.enrollment_step = 0
        self.enrollment_samples = []
        self.enroll_username_entry.clear()
        self.enroll_password_entry.clear()
        self.re_enroll_checkbox.setChecked(False)
        self.is_re_enrolling = False
        self.current_enroll_quote_data = random.choice(ENROLL_QUOTES) if ENROLL_QUOTES else "Please check enroll_quotes.csv"
        self.update_enroll_prompt()
        self.stacked_widget.setCurrentIndex(1)
        self._update_button_state(self.enroll_typing_entry, self.current_enroll_quote_data, self.enroll_submit_button)

    def go_to_login_page(self):
        self.current_session_id = f"sess-auth-{uuid.uuid4().hex[:12]}"
        self.current_session_token = uuid.uuid4().hex # Generate new single-use token
        self.status_widget.hide()
        self.login_input_widget.show()
        self.reset_keystroke_data()
        self.login_username_entry.clear()
        self.admin_username_entry.clear()
        self.admin_password_entry.clear()
        self.current_verify_quote_data = random.choice(VERIFY_QUOTES) if VERIFY_QUOTES else "Please check verify_quotes.csv"
        quote = self.current_verify_quote_data
        safe_quote = quote.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.login_quote_label.setText(f"<span style='{HIGHLIGHT_STYLE}'>{safe_quote}</span>")
        self.stacked_widget.setCurrentIndex(0)
        self._update_button_state(self.login_typing_entry, self.current_verify_quote_data, self.verify_button)
        self.dashboard_button.hide()

    def go_to_dashboard_page(self, data):
        self.health_gauge.setValue(data.get('health', 0))
        self.consistency_label.setText(f"Consistency: {data.get('consistency', 0):.1f}%")
        self.performance_label.setText(f"Performance: {data.get('performance', 0):.1f}%")
        self.score_chart.update_chart(data.get('scores', []))
        self.logins_value.setText(str(data.get('logins', 0)))
        self.drift_value.setText(f"{data.get('drift', 0)} / {DRIFT_SESSIONS_FOR_REANCHOR}")
        self.anomalies_value.setText(f"{data.get('anomalies', 0)} / {CONSECUTIVE_ANOMALY_LIMIT}")
        self.stacked_widget.setCurrentWidget(self.dashboard_widget)

    def show_dashboard_with_data(self):
        self.go_to_dashboard_page(self.dashboard_data)

    def reset_keystroke_data(self):
        self.completed_events.clear()
        self.press_times.clear()
        if hasattr(self, 'login_typing_entry'): self.login_typing_entry.clear()
        if hasattr(self, 'enroll_typing_entry'): self.enroll_typing_entry.clear()

    def update_enroll_prompt(self):
        self.enroll_prompt_label.setText(f"Sample {self.enrollment_step + 1}/{NUM_ENROLL_SAMPLES}:")
        quote = self.current_enroll_quote_data
        safe_quote = quote.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.enroll_quote_label.setText(f"<span style='{HIGHLIGHT_STYLE}'>{safe_quote}</span>")
        self.enroll_submit_button.setText("Create Profile" if self.enrollment_step >= NUM_ENROLL_SAMPLES - 1 else "Submit Sample")
        self._update_button_state(self.enroll_typing_entry, self.current_enroll_quote_data, self.enroll_submit_button)

    def toggle_free_type_mode(self, state):
        if state == Qt.CheckState.Checked.value:
            self.login_quote_label.hide()
            self.login_typing_entry.setPlaceholderText("Type 120-140 characters of any text...")
        else:
            self.login_quote_label.show()
            self.login_typing_entry.setPlaceholderText("Type the text shown above...")
        self.login_typing_entry.clear()
        self._update_button_state(self.login_typing_entry, self.current_verify_quote_data, self.verify_button)

    def _on_login_text_changed(self):
        if self.free_type_checkbox.isChecked():
            self._update_button_state(self.login_typing_entry, self.current_verify_quote_data, self.verify_button)
        else:
            self._update_highlight(self.login_typing_entry, self.login_quote_label, self.current_verify_quote_data, self.verify_button)

    def _on_enroll_text_changed(self):
        self._update_highlight(self.enroll_typing_entry, self.enroll_quote_label, self.current_enroll_quote_data, self.enroll_submit_button)

    def _update_button_state(self, typing_entry, quote, button):
        typed_length = len(typing_entry.toPlainText())
        is_login_page_free_type = (button == self.verify_button and hasattr(self, 'free_type_checkbox') and self.free_type_checkbox.isChecked())
        if is_login_page_free_type:
            button.setEnabled(typed_length >= 120)
        else:
            if not quote or not isinstance(quote, str): button.setEnabled(False); return
            quote_length = len(quote)
            button.setEnabled(typed_length >= quote_length * 0.93)

    def _update_highlight(self, typing_entry, quote_label, quote, button):
        typed_length = len(typing_entry.toPlainText())
        self._update_button_state(typing_entry, quote, button)
        if not quote or not isinstance(quote, str): return
        typed_length = min(typed_length, len(quote))
        dehighlighted_part, highlighted_part = quote[:typed_length], quote[typed_length:]
        dehighlighted_part_safe = dehighlighted_part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        highlighted_part_safe = highlighted_part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        new_text = (f"<span>{dehighlighted_part_safe}</span><span style='{HIGHLIGHT_STYLE}'>{highlighted_part_safe}</span>")
        quote_label.blockSignals(True)
        quote_label.setText(new_text)
        quote_label.blockSignals(False)

    def process_events_to_features(self, events):
        if len(events) < 2: return None
        return [[max(0, p_rel - p_pre), max(0, c_pre - p_pre), max(0, c_rel - p_pre), max(0, c_pre - p_rel), max(0, c_rel - p_rel)]
                for (_, p_pre, p_rel), (_, c_pre, c_rel) in zip(events, events[1:])]

    def get_encryption_key(self, username):
        secret = keyring.get_password(KEYRING_SERVICE_NAME, username)
        if not secret:
            secret = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
            keyring.set_password(KEYRING_SERVICE_NAME, username, secret)
        kdf = PBKDF2HMAC(hashes.SHA256(), 32, SECRET_DERIVATION_SALT, KDF_ITERATIONS, default_backend())
        return base64.urlsafe_b64encode(kdf.derive(secret.encode()))

    @staticmethod
    def hash_password(password):
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(hashes.SHA256(), 32, salt, KDF_ITERATIONS, default_backend())
        return kdf.derive(password.encode()).hex(), salt.hex()

    def verify_password(self, stored_hash_hex, salt_hex, provided_password):
        salt, stored_hash = bytes.fromhex(salt_hex), bytes.fromhex(stored_hash_hex)
        kdf = PBKDF2HMAC(hashes.SHA256(), 32, salt, KDF_ITERATIONS, default_backend())
        return kdf.derive(provided_password.encode()) == stored_hash

    def _calculate_template_health(self, rolling_window, recent_anchor_scores, baseline_variability=None):
        if len(rolling_window) < MIN_SAMPLES_FOR_DYNAMIC_THRESH:
            consistency_score = 100.0
        else:
            if baseline_variability is not None:
                personal_tight_bound = baseline_variability * 0.8
                personal_loose_bound = baseline_variability * 1.5
            else:
                personal_tight_bound = TIGHT_CONSISTENCY_STD_DEV
                personal_loose_bound = LOOSE_CONSISTENCY_STD_DEV

            consistency_metric = np.mean(np.std(rolling_window, axis=0))

            if consistency_metric <= personal_tight_bound:
                consistency_score = 100.0
            elif consistency_metric >= personal_loose_bound:
                consistency_score = 0.0
            else:
                dev_range = personal_loose_bound - personal_tight_bound
                if dev_range > 0:
                    normalized_dev = (consistency_metric - personal_tight_bound) / dev_range
                    consistency_score = 100 * (1 - normalized_dev)
                else:
                    consistency_score = 100.0

        if not recent_anchor_scores:
            performance_score = 100.0
        else:
            performance_score = np.mean(recent_anchor_scores) * 100

        health_score = (CONSISTENCY_WEIGHT * consistency_score) + (PERFORMANCE_WEIGHT * performance_score)
        return max(0, min(100, health_score)), max(0, min(100, consistency_score)), performance_score

    def extract_esn_features(self, sequence, mask):
        W_in, W_res, washout, leak_rate = MODEL['W_input'], MODEL['W_reservoir'], MODEL['washout_period'], MODEL['leak_rate']
        sequence_length = len(sequence)
        reservoir_size = W_res.shape[0]

        states = np.zeros((sequence_length, reservoir_size))

        for t in range(1, sequence_length):
            if mask[t]:
                u_t = np.array(sequence[t])
                input_contribution = W_in @ u_t
                reservoir_contribution = W_res @ states[t-1]
                new_activation = np.tanh(input_contribution + reservoir_contribution)
                states[t] = (1 - leak_rate) * states[t-1] + leak_rate * new_activation
            else:
                states[t] = states[t-1]

        post_washout_mask = mask[washout:]
        valid_states = states[washout:][post_washout_mask]

        return np.mean(valid_states, axis=0) if len(valid_states) > 0 else states[-1]

    def enroll_submit(self):
        if self.enrollment_step == 0:
            self.enroll_username = self.enroll_username_entry.text().strip()
            self.enroll_password = self.enroll_password_entry.text()
            if not self.enroll_username or not self.enroll_password:
                self.show_message_box("Input Error", "Username and password cannot be empty.", QMessageBox.Icon.Warning)
                return

            profile_exists = os.path.exists(os.path.join(TEMPLATE_DIR, f"{self.enroll_username}.dat"))
            is_re_enroll_checked = self.re_enroll_checkbox.isChecked()

            if profile_exists:
                if is_re_enroll_checked:
                    metadata = self._get_user_metadata(self.enroll_username)
                    if metadata and self.verify_password(metadata['password_hash'], metadata['salt'], self.enroll_password):
                        self.enrollment_samples.clear()
                        self.enrollment_step = 0
                        try:
                            dat_path_to_delete = os.path.join(TEMPLATE_DIR, f"{self.enroll_username}.dat")
                            hash_path_to_delete = os.path.join(TEMPLATE_DIR, f"{self.enroll_username}.hash")
                            if os.path.exists(dat_path_to_delete): os.remove(dat_path_to_delete)
                            if os.path.exists(hash_path_to_delete): os.remove(hash_path_to_delete)
                        except OSError as e:
                            self.show_message_box("File Error", f"Could not delete the old profile: {e}", QMessageBox.Icon.Critical)
                            return
                        self.is_re_enrolling = True
                    else:
                        self.show_message_box("Authorization Failed", "Incorrect password for this user. Cannot re-enroll.", QMessageBox.Icon.Critical)
                        return
                else:
                    self.show_message_box("Error", "This username is already taken. To overwrite, check the 'Re-enroll' box.", QMessageBox.Icon.Critical)
                    return
            elif is_re_enroll_checked:
                self.show_message_box("Error", f"Profile for '{self.enroll_username}' not found. Cannot re-enroll.", QMessageBox.Icon.Critical)
                return

        typed_text = self.enroll_typing_entry.toPlainText()
        quote_text = self.current_enroll_quote_data
        is_valid, error_message = self._validate_text_accuracy(typed_text, quote_text)
        if not is_valid:
            QMessageBox.warning(self, "Typing Accuracy Error", error_message); return

        timings = self.process_events_to_features(self.completed_events)
        if not timings:
            self.show_message_box("Processing Error", "Could not generate features from keystrokes.", QMessageBox.Icon.Warning); return

        self.enrollment_samples.append(timings)
        self.enrollment_step += 1
        self.reset_keystroke_data()

        if self.enrollment_step < NUM_ENROLL_SAMPLES:
            self.show_message_box("Sample Captured", f"Sample {self.enrollment_step}/{NUM_ENROLL_SAMPLES} captured successfully.")
            self.current_enroll_quote_data = random.choice(ENROLL_QUOTES) if ENROLL_QUOTES else "Check file."
            self.update_enroll_prompt()
        else:
            self.create_user_profile(self.enroll_username, self.enroll_password, self.enrollment_samples)
            msg = "Your profile has been re-enrolled successfully." if self.is_re_enrolling else "Your profile has been created successfully."
            self.show_message_box("Enrollment Complete", msg)
            self.go_to_login_page()

    def create_user_profile(self, username, password, all_samples):
        from itertools import combinations

        esn_vectors = []
        for s in all_samples:
            scaled_s = MODEL['input_scaler'].transform(np.array(s))
            mask = np.ones(len(scaled_s), dtype=bool)
            esn_features = self.extract_esn_features(scaled_s, mask)
            esn_vectors.append(MODEL['feature_scaler'].transform(esn_features.reshape(1, -1)).flatten())
        
        if len(esn_vectors) > 1:
            baseline_variability = np.mean(np.std(esn_vectors, axis=0))
        else:
            baseline_variability = 0.20
        
        svm_scores, cos_sims, euc_dists = [], [], []
        for v1, v2 in combinations(esn_vectors, 2):
            svm_scores.append(MODEL['svm_classifier'].predict_proba(np.abs(v1 - v2).reshape(1, -1))[0, 1])
            cos_sims.append(cosine_similarity([v1], [v2])[0, 0])
            euc_dists.append(euclidean_distances([v1], [v2])[0, 0])
        password_hash, salt = self.hash_password(password)
        statistical_template = np.mean(np.vstack([np.array(s) for s in all_samples]), axis=0)
        user_id = hashlib.sha3_256(username.encode()).hexdigest()[:16]

        # THRESHOLD LOGIC
        metadata = {"user_id": user_id, "username": username,
                    "svm_threshold": max(MIN_SECURE_SVM_THRESHOLD, (np.percentile(svm_scores, 5) - 0.08) if svm_scores else MIN_SECURE_SVM_THRESHOLD),
                    "cosine_threshold": (np.percentile(cos_sims, 5) - 0.02) if cos_sims else MIN_SECURE_COSINE_THRESHOLD,
                    "distance_threshold": (np.percentile(euc_dists, 95) + 1.0) if euc_dists else 10.0,
                    "password_hash": password_hash, "salt": salt, "drift_counter": 0,
                    "first_login_pending": True,
                    "login_count": 0,
                    "recent_anchor_scores": [],
                    "proactive_snooze_until": 0,
                    "baseline_variability": float(baseline_variability), 
                    "consecutive_anomaly_count": 0
                    }

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('metadata.json', json.dumps(metadata))
            npz_buffer = io.BytesIO()
            np.savez(npz_buffer,
                     esn_anchor=np.mean(esn_vectors, axis=0),
                     statistical_template=statistical_template,
                     rolling_window=np.array([]),
                     quarantined_samples=np.array([]))
            zf.writestr('template.npz', npz_buffer.getvalue())
        encrypted_data = Fernet(self.get_encryption_key(username)).encrypt(zip_buffer.getvalue())
        dat_path = os.path.join(TEMPLATE_DIR, f"{username}.dat")
        with open(dat_path, 'wb') as f: f.write(encrypted_data)
        with open(dat_path.replace('.dat', '.hash'), 'w') as f: f.write(hashlib.sha3_256(encrypted_data).hexdigest())

        log_event = {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "user_id": metadata["user_id"],
            "username": username,
            "samples": NUM_ENROLL_SAMPLES,
            "thresholds": f"svm:{metadata['svm_threshold']:.2f} cos:{metadata['cosine_threshold']:.2f} euc:{metadata['distance_threshold']:.1f}"
        }
        if self.is_re_enrolling:
            log_event["event_type"] = "REENROLL_SUCCESS"
            log_event["session_id"] = self.current_session_id
            log_event["password_verified"] = True
        else:
            log_event["event_type"] = "ENROLL_SUCCESS"
            log_event["session_id"] = self.current_session_id
        self.logger.log_event(log_event)

    def is_text_linguistically_valid(self, text):
        """Checks if the input text is likely real language using the loaded dictionary."""
        if not DICTIONARY_WORDS:
            # Fallback in case dictionary failed to load. This prevents the app from being
            # unusable if the dictionary file is missing post-startup.
            print("Warning: Dictionary not loaded, cannot perform gibberish check.")
            return True, 100.0

        # Sanitize and split text into words, removing punctuation.
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return False, 0.0

        valid_word_count = sum(1 for word in words if word in DICTIONARY_WORDS)
        validity_ratio = (valid_word_count / len(words)) * 100.0

        is_valid = validity_ratio >= GIBBERISH_VALIDITY_THRESHOLD
        return is_valid, validity_ratio

    # MODIFIED METHOD
    def verify_submit(self):
        if self.current_session_token is None:
            self.show_message_box("Session Error", "This login session has expired or is invalid. Please try again.", QMessageBox.Icon.Warning)
            self.go_to_login_page()
            return

        self.current_session_token = None

        username = self.login_username_entry.text().strip()
        if not username:
            self.show_message_box("Input Error", "Please provide a username.", QMessageBox.Icon.Warning); return

        now = time.time()
        if username in self.failed_attempts:
            self.failed_attempts[username] = [t for t in self.failed_attempts[username] if now - t < LOCKOUT_PERIOD_SECONDS]

            if len(self.failed_attempts[username]) >= MAX_FAILED_ATTEMPTS:
                oldest_fail_time = self.failed_attempts[username][0]
                time_left = int(LOCKOUT_PERIOD_SECONDS - (now - oldest_fail_time))
                self.show_message_box("Account Locked",
                                      f"Too many failed attempts. Please try again in {time_left // 60} minutes and {time_left % 60} seconds.",
                                      QMessageBox.Icon.Critical)
                return

        dat_path = os.path.join(TEMPLATE_DIR, f"{username}.dat")
        if not os.path.exists(dat_path):
            self.show_message_box("Authentication Failed", "User not found. Please check the username or enroll.", QMessageBox.Icon.Warning)
            self.logger.log_event({
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_type": "AUTH_FAIL", "username": username, "error": "UserNotFound",
                "session_id": self.current_session_id
            })
            return

        typed_text = self.login_typing_entry.toPlainText()

        # Text validation logic
        if self.free_type_checkbox.isChecked():
            # For Free Type Mode, perform the gibberish check
            is_linguistically_valid, validity_ratio = self.is_text_linguistically_valid(typed_text)
            if not is_linguistically_valid:
                self.show_message_box(
                    "Authentication Failed",
                    "Gibberish input was rejected. Please type meaningful text to verify.",
                    QMessageBox.Icon.Warning
                )
                self.logger.log_event({
                    "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "event_type": "AUTH_FAIL",
                    "username": username,
                    "session_id": self.current_session_id,
                    "error": "GibberishInputRejected",
                    "details": f"Word validity ratio was {validity_ratio:.1f}%, which is below the {GIBBERISH_VALIDITY_THRESHOLD:.1f}% threshold."
                })
                # It's a failed attempt, so update the rate limiter
                if username not in self.failed_attempts: self.failed_attempts[username] = []
                self.failed_attempts[username].append(time.time())
                self.go_to_login_page() # Reset for another attempt
                return
        else:
            # For Quote Mode, perform the accuracy check
            quote_text = self.current_verify_quote_data
            is_valid, error_message = self._validate_text_accuracy(typed_text, quote_text)
            if not is_valid:
                QMessageBox.warning(self, "Typing Accuracy Error", error_message)
                return

        live_timings = self.process_events_to_features(self.completed_events)
        if not live_timings:
            self.show_message_box("Processing Error", "Could not generate features from keystrokes.", QMessageBox.Icon.Warning); return

        verification_result = self.verify_user(username, live_timings)
        if not verification_result:
            # Error messages are shown within verify_user if profile is incompatible
            self.show_rejection_screen("Profile Error")
            self.logger.log_event({
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_type": "AUTH_FAIL", "username": username, "error": "ProfileCorruptOrUnreadable",
                "session_id": self.current_session_id
            })
            return

        metadata = verification_result['metadata']
        scores = verification_result['scores']
        is_adaptive_match = verification_result['is_adaptive_match']
        is_anchor_match = verification_result['is_anchor_match']

        if scores['svm_adaptive'] >= SUSPICIOUS_SCORE_THRESHOLD or scores['svm_anchor'] >= SUSPICIOUS_SCORE_THRESHOLD:
            is_authenticated = is_adaptive_match or is_anchor_match
        else:
            is_authenticated = is_adaptive_match and is_anchor_match

        auth_override_reason = ""
        if is_authenticated and (scores['svm_adaptive'] < CONFIDENCE_FLOOR and scores['svm_anchor'] < CONFIDENCE_FLOOR):
            is_authenticated = False
            auth_override_reason = "OVERRIDE_FAIL(ConfidenceTooLow)"

        self.last_auth_was_success = is_authenticated
        
        if is_authenticated:
            if username in self.failed_attempts:
                del self.failed_attempts[username]
        else:
            if username not in self.failed_attempts:
                self.failed_attempts[username] = []
            self.failed_attempts[username].append(time.time())

        self.login_input_widget.hide()
        self.status_widget.show()

        if is_authenticated:
            is_suspicious = scores['svm_adaptive'] < SUSPICIOUS_SCORE_THRESHOLD and scores['svm_anchor'] < SUSPICIOUS_SCORE_THRESHOLD
            is_first_login = metadata.get("first_login_pending", False)

            if is_suspicious and not is_first_login:
                step_up_dialog = StepUpDialog(self)
                if step_up_dialog.exec() == QDialog.DialogCode.Accepted:
                    password = step_up_dialog.get_password()
                    if self.verify_password(metadata['password_hash'], metadata['salt'], password):
                        self.logger.log_event({"event_type": "STEP_UP_SUCCESS", "user_id": metadata.get("user_id", ""), "username": username, "session_id": self.current_session_id})
                        self.show_success_screen("Authenticated")
                        self.dashboard_button.show()
                    else:
                        is_authenticated = False
                        self.logger.log_event({"event_type": "STEP_UP_FAIL", "user_id": metadata.get("user_id", ""), "username": username, "session_id": self.current_session_id})
                        self.show_rejection_screen("Rejected")
                        self.dashboard_button.hide()
                else:
                    is_authenticated = False
                    self.logger.log_event({"event_type": "STEP_UP_CANCEL", "user_id": metadata.get("user_id", ""), "username": username, "session_id": self.current_session_id})
                    self.show_rejection_screen("Cancelled")
                    self.dashboard_button.hide()
            else:
                self.show_success_screen("Authenticated")
                self.dashboard_button.show()
        else:
            self.show_rejection_screen("Rejected")
            self.dashboard_button.hide()

        drift_counter = metadata.get('drift_counter', 0)
        next_drift_counter = drift_counter
        if is_authenticated:
            if is_adaptive_match and not is_anchor_match:
                next_drift_counter = drift_counter + 1
            else:
                next_drift_counter = 0

        primary_res = "PASS" if is_adaptive_match else "FAIL"
        safety_res = "PASS" if is_anchor_match else "FAIL"
        auth_results_str = (f"ADAPTIVE_{primary_res}(svm:{scores['svm_adaptive']:.2f},cos:{scores['cos_adaptive']:.2f},euc:{scores['euc_adaptive']:.1f}) | "
                            f"ANCHOR_{safety_res}(svm:{scores['svm_anchor']:.2f},cos:{scores['cos_anchor']:.2f},euc:{scores['euc_anchor']:.1f})")
        if auth_override_reason:
            auth_results_str += f" | {auth_override_reason}"

        log_data = {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_type": "AUTH_SUCCESS" if is_authenticated else "AUTH_FAIL",
            "user_id": metadata.get("user_id", ""),
            "username": username,
            "session_id": self.current_session_id,
            "auth_results": auth_results_str,
            "baseline_thresholds": f"svm:{metadata['svm_threshold']:.2f} cos:{metadata['cosine_threshold']:.2f} euc:{metadata['distance_threshold']:.1f}",
            "drift": f"{next_drift_counter}/{DRIFT_SESSIONS_FOR_REANCHOR}",
            "threshold_mode": verification_result.get("threshold_mode", "NORMAL"),
            "method": "ADAPTIVE" if is_adaptive_match else ("ANCHOR" if is_anchor_match else "NONE"),
            "typing_pattern": verification_result.get("typing_pattern", "unknown")
        }
        self.logger.log_event(log_data)

        if is_authenticated:
            self.dashboard_data = self.save_user_profile(username, verification_result, next_drift_counter)

    def verify_user(self, username, timings):
        dat_path = os.path.join(TEMPLATE_DIR, f"{username}.dat")
        try:
            with open(dat_path, 'rb') as f: encrypted_data = f.read()
            hash_path = dat_path.replace('.dat', '.hash')
            if not os.path.exists(hash_path) or hashlib.sha3_256(encrypted_data).hexdigest() != open(hash_path, 'r').read():
                self.logger.log_event({"event_type": "TAMPER_ALERT", "file": f"{username}.dat"})
                return None

            decrypted_data = Fernet(self.get_encryption_key(username)).decrypt(encrypted_data)
            with zipfile.ZipFile(io.BytesIO(decrypted_data), 'r') as zf:
                metadata = json.loads(zf.read('metadata.json'))
                with io.BytesIO(zf.read('template.npz')) as npz_buffer:
                    npz_files = np.load(npz_buffer, allow_pickle=True)
                    esn_anchor = npz_files['esn_anchor']
                    statistical_template = npz_files['statistical_template']
                    rolling_window = list(npz_files['rolling_window']) if 'rolling_window' in npz_files and npz_files['rolling_window'].size > 0 else []
                    quarantined_samples = list(npz_files['quarantined_samples']) if 'quarantined_samples' in npz_files and npz_files['quarantined_samples'].size > 0 else []
        except Exception:
            return None

        scaled_timings = MODEL['input_scaler'].transform(np.array(timings))
        mask = np.ones(len(scaled_timings), dtype=bool)
        esn_features = self.extract_esn_features(scaled_timings, mask)
        new_feature_vector = MODEL['feature_scaler'].transform(esn_features.reshape(1, -1)).flatten()

        if new_feature_vector.shape[0] != esn_anchor.shape[0]:
            self.logger.log_event({
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_type": "AUTH_FAIL", "username": username, "error": "ProfileModelMismatch",
                "details": f"Live vector shape: {new_feature_vector.shape}, Stored anchor shape: {esn_anchor.shape}"
            })
            self.show_message_box("Profile Outdated",
                                  "Your profile is incompatible with the current system model. "
                                  "Please use the 'Re-enroll' option on the enrollment page to update your profile.",
                                  QMessageBox.Icon.Critical)
            return None
        base_svm_thresh, base_cos_thresh, base_dist_thresh = metadata['svm_threshold'], metadata['cosine_threshold'], metadata['distance_threshold']
        dynamic_svm_thresh, dynamic_cos_thresh, dynamic_dist_thresh = base_svm_thresh, base_cos_thresh, base_dist_thresh
        threshold_mode = "NORMAL"

        if len(rolling_window) >= MIN_SAMPLES_FOR_DYNAMIC_THRESH:
            consistency_metric = np.mean(np.std(rolling_window, axis=0))
            if consistency_metric < TIGHT_CONSISTENCY_STD_DEV:
                dynamic_svm_thresh *= 1.05; dynamic_cos_thresh *= 1.05; dynamic_dist_thresh *= 0.95
                threshold_mode = "STRICT"
            elif consistency_metric > LOOSE_CONSISTENCY_STD_DEV:
                dynamic_svm_thresh *= 0.85; dynamic_cos_thresh *= 0.95; dynamic_dist_thresh *= 2
                threshold_mode = "LENIENT"

        adaptive_template = (ANCHOR_WEIGHT * esn_anchor) + (WINDOW_WEIGHT * np.mean(rolling_window, axis=0)) if rolling_window else esn_anchor
        svm_adaptive = MODEL['svm_classifier'].predict_proba(np.abs(new_feature_vector - adaptive_template).reshape(1, -1))[0, 1]
        cos_adaptive = cosine_similarity([new_feature_vector], [adaptive_template])[0, 0]
        euc_adaptive = euclidean_distances([new_feature_vector], [adaptive_template])[0, 0]
        is_adaptive_match = (svm_adaptive >= dynamic_svm_thresh and cos_adaptive >= dynamic_cos_thresh and euc_adaptive <= dynamic_dist_thresh)

        svm_anchor_score = MODEL['svm_classifier'].predict_proba(np.abs(new_feature_vector - esn_anchor).reshape(1, -1))[0, 1]
        cos_anchor_score = cosine_similarity([new_feature_vector], [esn_anchor])[0, 0]
        euc_anchor_score = euclidean_distances([new_feature_vector], [esn_anchor])[0, 0]
        is_anchor_match = (svm_anchor_score >= base_svm_thresh and cos_anchor_score >= base_cos_thresh and euc_anchor_score <= base_dist_thresh)

        return {
            "metadata": metadata,
            "esn_anchor": esn_anchor,
            "statistical_template": statistical_template,
            "rolling_window": rolling_window,
            "new_feature_vector": new_feature_vector,
            "is_adaptive_match": is_adaptive_match,
            "is_anchor_match": is_anchor_match,
            "quarantined_samples": quarantined_samples,
            "scores": {
                'svm_adaptive': float(svm_adaptive), 'cos_adaptive': float(cos_adaptive), 'euc_adaptive': float(euc_adaptive),
                'svm_anchor': float(svm_anchor_score), 'cos_anchor': float(cos_anchor_score), 'euc_anchor': float(euc_anchor_score)
            },
            "threshold_mode": threshold_mode,
            "baseline_variability": metadata.get("baseline_variability", None),
            "typing_pattern": self._get_typing_pattern(timings, statistical_template)
        }

    def save_user_profile(self, username, verification_result, drift_counter):
        metadata = verification_result['metadata']
        esn_anchor = verification_result['esn_anchor']
        statistical_template = verification_result['statistical_template']
        rolling_window = verification_result['rolling_window']
        quarantined_samples = verification_result['quarantined_samples']
        new_sample = verification_result['new_feature_vector']
        scores = verification_result['scores']
        is_anchor_match = verification_result['is_anchor_match']
        baseline_variability = verification_result.get('baseline_variability')

        sample_disposition = "DISCARDED"
        if scores['svm_adaptive'] >= SUSPICIOUS_SCORE_THRESHOLD and is_anchor_match:
            sample_disposition = "TRUSTED"
            rolling_window.append(new_sample)
            if quarantined_samples:
                rolling_window.extend(quarantined_samples)
                quarantined_samples.clear()
            metadata['consecutive_anomaly_count'] = 0
        elif scores['svm_adaptive'] >= SUSPICIOUS_SCORE_THRESHOLD and not is_anchor_match:
            sample_disposition = "DRIFT"
            rolling_window.append(new_sample)
            metadata['consecutive_anomaly_count'] = 0
        else:
            sample_disposition = "QUARANTINED"
            quarantined_samples.append(new_sample)
            metadata['consecutive_anomaly_count'] = metadata.get('consecutive_anomaly_count', 0) + 1
        self.logger.log_event({"timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "event_type": "SAMPLE_FILTER", "disposition": sample_disposition, "user_id": metadata.get('user_id'), "username": username, "anomaly_count": metadata.get('consecutive_anomaly_count', 0)})

        if metadata.get('consecutive_anomaly_count', 0) >= CONSECUTIVE_ANOMALY_LIMIT and quarantined_samples:
            dialog = MandatoryReAnchorDialog(self)
            password_verified = False
            if dialog.exec() == QDialog.DialogCode.Accepted:
                password = dialog.get_password()
                if self.verify_password(metadata['password_hash'], metadata['salt'], password):
                    password_verified = True

            if password_verified:
                esn_anchor = np.mean(quarantined_samples, axis=0)
                self.logger.log_event({"timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "event_type": "REANCHOR_SUCCESS", "reason": "PERSISTENT_ANOMALY", "user_id": metadata.get('user_id'), "username": username, "samples_used": len(quarantined_samples)})
                rolling_window, quarantined_samples, drift_counter = [], [], 0
                metadata['consecutive_anomaly_count'] = 0
                metadata['recent_anchor_scores'] = []
                self.show_message_box("Profile Secured", "Your biometric anchor has been updated based on your recent typing.", QMessageBox.Icon.Information)
            else:
                self.show_message_box("Update Failed", "Incorrect password. Profile not updated.", QMessageBox.Icon.Warning)

        if metadata.get("first_login_pending", False): metadata["first_login_pending"] = False
        login_count = metadata.get('login_count', 0) + 1
        metadata['login_count'] = login_count
        recent_scores = metadata.get('recent_anchor_scores', [])
        recent_scores.append(scores['svm_anchor'])
        metadata['recent_anchor_scores'] = recent_scores[-MAX_WINDOW_SIZE:]

        if drift_counter >= DRIFT_SESSIONS_FOR_REANCHOR and rolling_window:
            if self.re_anchor_prompt(username, metadata):
                esn_anchor = np.mean(rolling_window, axis=0)
                self.logger.log_event({"timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "event_type": "REANCHOR_SUCCESS", "reason": "DRIFT", "user_id": metadata.get('user_id'), "username": username, "session_id": self.current_session_id})
                rolling_window, drift_counter, quarantined_samples = [], 0, []
                metadata['recent_anchor_scores'] = []
                metadata['consecutive_anomaly_count'] = 0
                self.show_message_box("Profile Updated", "Your biometric anchor has been successfully updated.")

        proactive_snooze_until = metadata.get('proactive_snooze_until', 0)
        if login_count >= PROACTIVE_MIN_SAMPLES and login_count > proactive_snooze_until:
            health_score, _, _ = self._calculate_template_health(rolling_window, metadata['recent_anchor_scores'], baseline_variability)
            if health_score < PROACTIVE_HEALTH_THRESHOLD:
                self.logger.log_event({"timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "event_type": "PROACTIVE_PROMPT_TRIGGERED", "health_score": f"{health_score:.1f}", "user_id": metadata.get('user_id'), "username": username})
                dialog = ProactiveReAnchorDialog(self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    if self.re_anchor_prompt(username, metadata) and rolling_window:
                        esn_anchor = np.mean(rolling_window, axis=0)
                        self.logger.log_event({"timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "event_type": "REANCHOR_SUCCESS", "reason": "PROACTIVE", "user_id": metadata.get('user_id'), "username": username})
                        rolling_window, drift_counter, quarantined_samples = [], 0, []
                        metadata['recent_anchor_scores'] = []
                        metadata['consecutive_anomaly_count'] = 0
                        self.show_message_box("Profile Updated", "Your biometric anchor has been successfully updated.")
                else:
                    metadata['proactive_snooze_until'] = login_count + PROACTIVE_SNOOZE_SESSIONS
                    self.logger.log_event({"timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "event_type": "PROACTIVE_PROMPT_SNOOZED", "user_id": metadata.get('user_id'), "username": username, "snooze_until": metadata['proactive_snooze_until']})

        metadata['drift_counter'] = drift_counter
        rolling_window = rolling_window[-MAX_WINDOW_SIZE:]
        quarantined_samples = quarantined_samples[-MAX_QUARANTINE_SIZE:] 

        health, consistency, performance = self._calculate_template_health(rolling_window, metadata['recent_anchor_scores'], baseline_variability)
        
        dashboard_payload = {
            'health': health,
            'consistency': consistency,
            'performance': performance,
            'scores': metadata['recent_anchor_scores'],
            'logins': metadata['login_count'],
            'drift': metadata['drift_counter'],
            'anomalies': metadata['consecutive_anomaly_count']
        }

        dat_path = os.path.join(TEMPLATE_DIR, f"{username}.dat")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('metadata.json', json.dumps(metadata))
            npz_buffer = io.BytesIO()
            np.savez(npz_buffer,
                     esn_anchor=esn_anchor,
                     statistical_template=statistical_template,
                     rolling_window=np.array(rolling_window),
                     quarantined_samples=np.array(quarantined_samples))
            zf.writestr('template.npz', npz_buffer.getvalue())

        new_encrypted_data = Fernet(self.get_encryption_key(username)).encrypt(zip_buffer.getvalue())
        with open(dat_path, 'wb') as f: f.write(new_encrypted_data)
        with open(dat_path.replace('.dat', '.hash'), 'w') as f: f.write(hashlib.sha3_256(new_encrypted_data).hexdigest())

        return dashboard_payload

    def admin_login(self):
        username_attempt = self.admin_username_entry.text().strip()
        password_attempt = self.admin_password_entry.text()
        session_id = f"sess-admin-{uuid.uuid4().hex[:8]}"

        try:
            with open(ADMIN_CONFIG_PATH, 'r') as f:
                admin_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.show_message_box("Admin Error", "Admin configuration is missing or corrupt. Please restart the application.", QMessageBox.Icon.Critical)
            return

        is_username_correct = (username_attempt == admin_config.get("username"))
        is_password_correct = self.verify_password(
            admin_config.get("password_hash"),
            admin_config.get("salt"),
            password_attempt
        )

        if is_username_correct and is_password_correct:
            self.logger.log_event({
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_type": "ADMIN_SUCCESS",
                "user_id": "admin", "username": "admin",
                "session_id": session_id,
                "action": "VIEW_LOGS"
            })
            log_dialog = LogViewerDialog(self.logger.read_logs(), self)
            log_dialog.exec()
        else:
            self.logger.log_event({
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_type": "ADMIN_FAIL",
                "user_id": username_attempt if username_attempt else "unknown",
                "username": username_attempt if username_attempt else "unknown",
                "session_id": session_id,
                "error": "INVALID_CREDENTIALS"
            })
            self.show_message_box("Admin Access Denied", "Incorrect username or password.", QMessageBox.Icon.Warning)
        
        self.admin_password_entry.clear()


# STARTUP
def initial_admin_setup():
    """Checks if admin is set up and prompts for creation if not."""
    if not os.path.exists(ADMIN_CONFIG_PATH):
        dialog = QDialog()
        dialog.setWindowTitle("Admin Setup Required")
        layout = QVBoxLayout(dialog)
        
        label = QLabel("This is the first run. Please create a password for the 'admin' user.")
        pwd_entry = QLineEdit()
        pwd_entry.setEchoMode(QLineEdit.EchoMode.Password)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        pwd_entry.textChanged.connect(lambda text: buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(text)))
        
        buttons.accepted.connect(dialog.accept)

        layout.addWidget(label)
        layout.addWidget(QLabel("New Admin Password:"))
        layout.addWidget(pwd_entry)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            password = pwd_entry.text()
            password_hash, salt = KeystrokeApp.hash_password(password)
            
            admin_config = {
                "username": "admin",
                "password_hash": password_hash,
                "salt": salt
            }
            with open(ADMIN_CONFIG_PATH, 'w') as f:
                json.dump(admin_config, f)
            QMessageBox.information(None, "Setup Complete", "Admin user has been configured.")
        else:
            sys.exit(0)

if __name__ == "__main__":
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    
    # Run the secure admin setup check on startup
    initial_admin_setup()

    try:
        if not all([load_model(),
                    load_quotes_from_csv(ENROLL_QUOTES_PATH, ENROLL_QUOTES),
                    load_quotes_from_csv(VERIFY_QUOTES_PATH, VERIFY_QUOTES, is_verify_list=True),
                    load_dictionary(DICTIONARY_PATH, DICTIONARY_WORDS)]):
            sys.exit(1)
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"A critical error occurred on startup: {e}")
        sys.exit(1)

    window = KeystrokeApp()
    window.show()
    sys.exit(app.exec())