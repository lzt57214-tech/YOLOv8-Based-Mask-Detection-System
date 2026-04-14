from __future__ import annotations
"""Simplified GUI for mini_yolo_app.

中文: mini_yolo_app 的精简图形化界面，仅保留图片检测和视频检测。
English: Simplified GUI with image and video detection only.
"""

from pathlib import Path

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "trained" / "best.pt"
OUTPUT_DIR = ROOT / "outputs" / "gui"


class MainWindow(QTabWidget):
    """Main GUI window.

    中文: 图形化主窗口。
    English: Main window for GUI interactions.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLOv8 Detection")
        self.resize(1100, 760)

        self.model = self._load_model(self._resolve_model_path())
        self.image_path: str = ""
        self.video_path: str = ""
        self.capture: cv2.VideoCapture | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_video_frame)

        self._build_ui()

    @staticmethod
    def _resolve_model_path() -> Path:
        """Pick the newest trained checkpoint.

        中文: 优先加载 outputs 下最新生成的 best.pt，避免 GUI 误用旧权重。
        English: Prefer the newest best.pt under outputs to avoid stale weights.
        """
        output_root = ROOT / "outputs"
        candidates = [p for p in output_root.rglob("best.pt") if p.is_file()]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        if DEFAULT_MODEL.exists():
            return DEFAULT_MODEL
        raise FileNotFoundError(f"Model not found: {DEFAULT_MODEL}")

    @staticmethod
    def _load_model(model_path: Path) -> YOLO:
        """Load fixed detection model.

        中文: 加载固定的检测模型权重。
        English: Load the fixed model checkpoint used by GUI.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return YOLO(str(model_path))

    @staticmethod
    def _to_pixmap(frame_bgr) -> QPixmap:
        """Convert OpenCV frame to Qt pixmap.

        中文: 将 OpenCV 的 BGR 图像转换为 Qt 可显示的 Pixmap。
        English: Convert OpenCV BGR frame to a Qt displayable pixmap.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    @staticmethod
    def _class_stats(result) -> str:
        """Build class count summary from one prediction result.

        中文: 根据单张推理结果统计各类别数量。
        English: Summarize per-class counts from one prediction result.
        """
        names = result.names
        nums = [0 for _ in range(len(names))]
        for cls_id in result.boxes.cls.cpu().numpy().tolist():
            nums[int(cls_id)] += 1
        lines = [f"{names[i]}: {n}" for i, n in enumerate(nums) if n > 0]
        return "\n".join(lines) if lines else "No detections"

    def _build_ui(self) -> None:
        """Create tab layout.

        中文: 创建两页签界面布局。
        English: Build two-tab user interface layout.
        """
        self.addTab(self._build_image_tab(), "Image Detection")
        self.addTab(self._build_video_tab(), "Video Detection")

    def _build_image_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.image_input = QLabel("Image to upload")
        self.image_output = QLabel("Detection result")
        self.image_input.setAlignment(Qt.AlignCenter)
        self.image_output.setAlignment(Qt.AlignCenter)
        self.image_input.setMinimumSize(480, 640)
        self.image_output.setMinimumSize(480, 640)
        row.addWidget(self.image_input)
        row.addWidget(self.image_output)

        self.image_info = QLabel("Detection status: waiting")

        upload_btn = QPushButton("Upload Image")
        detect_btn = QPushButton("Run Detection")
        upload_btn.clicked.connect(self._upload_image)
        detect_btn.clicked.connect(self._detect_image)

        layout.addLayout(row)
        layout.addWidget(self.image_info)
        layout.addWidget(upload_btn)
        layout.addWidget(detect_btn)
        tab.setLayout(layout)
        return tab

    def _build_video_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        self.video_view = QLabel("Video to upload")
        self.video_view.setAlignment(Qt.AlignCenter)
        self.video_view.setMinimumSize(960, 640)
        self.video_info = QLabel("Detection status: waiting")

        open_btn = QPushButton("Upload Video")
        start_btn = QPushButton("Start Detection")
        stop_btn = QPushButton("Stop Detection")
        open_btn.clicked.connect(self._open_video)
        start_btn.clicked.connect(self._start_video_detection)
        stop_btn.clicked.connect(self._stop_video_detection)

        layout.addWidget(self.video_view)
        layout.addWidget(self.video_info)
        layout.addWidget(open_btn)
        layout.addWidget(start_btn)
        layout.addWidget(stop_btn)
        tab.setLayout(layout)
        return tab

    def _upload_image(self) -> None:
        """Select and preview an image file.

        中文: 选择并预览待检测图片。
        English: Choose and preview image for detection.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "*.jpg *.jpeg *.png *.bmp")
        if not file_name:
            return

        self.image_path = file_name
        frame = cv2.imread(file_name)
        if frame is None:
            QMessageBox.warning(self, "Error", "Failed to read image")
            return

        self.image_input.setPixmap(self._to_pixmap(frame).scaled(480, 640, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.image_info.setText("Detection status: waiting")

    def _detect_image(self) -> None:
        """Run image inference and display/save result.

        中文: 执行图片检测并显示/保存结果。
        English: Run image detection and display/save output.
        """
        if not self.image_path:
            QMessageBox.information(self, "Info", "Please upload an image first")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results = self.model(self.image_path, conf=0.25, iou=0.45)
        result = results[0]
        plotted = result.plot()
        save_path = OUTPUT_DIR / "image_result.jpg"
        cv2.imwrite(str(save_path), plotted)
        self.image_output.setPixmap(self._to_pixmap(plotted).scaled(480, 640, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.image_info.setText(f"Detection result:\n{self._class_stats(result)}")

    def _open_video(self) -> None:
        """Select local video file.

        中文: 选择本地视频文件。
        English: Choose local video source.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose video", "", "*.mp4 *.avi *.mov *.mkv")
        if not file_name:
            return
        self.video_path = file_name
        self.video_info.setText("Detection status: waiting")

    def _start_video_detection(self) -> None:
        """Open selected video and start timer loop.

        中文: 打开所选视频并开启定时推理循环。
        English: Open selected video and start timed inference loop.
        """
        if not self.video_path:
            QMessageBox.information(self, "Info", "Please upload a video first")
            return

        self._stop_video_detection()
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            QMessageBox.warning(self, "Error", "Failed to open video")
            self.capture = None
            return

        self.timer.start(30)

    def _process_video_frame(self) -> None:
        """Process one frame on each timer tick.

        中文: 每次定时器触发处理一帧视频。
        English: Process one video frame per timer tick.
        """
        if self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self._stop_video_detection()
            return

        result = self.model(frame, conf=0.25, iou=0.45)[0]
        plotted = result.plot()
        self.video_view.setPixmap(self._to_pixmap(plotted).scaled(960, 640, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.video_info.setText(f"Detection result:\n{self._class_stats(result)}")

    def _stop_video_detection(self) -> None:
        """Stop timer and release video resource.

        中文: 停止检测并释放视频资源。
        English: Stop detection loop and release video capture.
        """
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def closeEvent(self, event) -> None:
        """Ensure resources are released on window close.

        中文: 关闭窗口时确保资源已释放。
        English: Ensure resources are released when window closes.
        """
        self._stop_video_detection()
        super().closeEvent(event)




