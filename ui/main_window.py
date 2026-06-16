"""
UI界面组件模块 - YOLOv11目标检测系统
"""

import sys
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QGroupBox, QGridLayout, 
                             QSlider, QSpinBox, QProgressBar, QSplitter,
                             QTextEdit, QCheckBox, QFrame, QFileDialog,
                             QMessageBox, QApplication, QMainWindow)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QImage, QPixmap


class ModernButton(QPushButton):
    """现代化按钮样式"""
    
    def __init__(self, text, parent=None, primary=False, danger=False):
        super().__init__(text, parent)
        self.primary = primary
        self.danger = danger
        self.setup_style()
        
    def setup_style(self):
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #6366f1;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #818cf8;
                }
                QPushButton:pressed {
                    background-color: #4f46e5;
                }
                QPushButton:disabled {
                    background-color: #374151;
                    color: #6b7280;
                }
            """)
        elif self.danger:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #f87171;
                }
                QPushButton:pressed {
                    background-color: #dc2626;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #374151;
                    color: #f3f4f6;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #4b5563;
                }
                QPushButton:pressed {
                    background-color: #1f2937;
                }
                QPushButton:disabled {
                    background-color: #1f2937;
                    color: #6b7280;
                }
            """)


class ModernSlider(QSlider):
    """现代化滑块控件"""
    
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #374151;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #6366f1;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #818cf8;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #a5b4fc;
            }
        """)


class ModernGroupBox(QGroupBox):
    """现代化分组框"""
    
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                color: #f3f4f6;
                border: 1px solid #374151;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 10px;
                font-size: 14px;
                font-weight: bold;
                background-color: #1f2937;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                background-color: #1f2937;
                color: #818cf8;
            }
        """)


class ImageDisplayLabel(QLabel):
    """图像显示标签，支持拖放"""
    
    # 定义信号
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(400)
        self.setStyleSheet("""
            QLabel {
                background-color: #111827;
                border: 2px dashed #374151;
                border-radius: 12px;
                color: #6b7280;
                font-size: 14px;
            }
            QLabel:hover {
                border-color: #6366f1;
                background-color: #1a2538;
            }
        """)
        self.setText("📷 拖拽图片或视频到此处\n\n或使用左侧按钮选择文件")
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                QLabel {
                    background-color: #1a2538;
                    border: 2px solid #6366f1;
                    border-radius: 12px;
                    color: #818cf8;
                }
            """)
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                background-color: #111827;
                border: 2px dashed #374151;
                border-radius: 12px;
                color: #6b7280;
                font-size: 14px;
            }
        """)
        
    def dropEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                background-color: #111827;
                border: 2px dashed #374151;
                border-radius: 12px;
                color: #6b7280;
                font-size: 14px;
            }
        """)
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)


class ControlPanel(QWidget):
    """控制面板组件"""
    
    # 信号定义
    load_model_clicked = pyqtSignal(str)
    load_image_clicked = pyqtSignal()
    load_video_clicked = pyqtSignal()
    toggle_camera_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    params_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("YOLOv11 检测系统")
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: #818cf8;
            padding: 10px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 模型配置
        self.model_group = ModernGroupBox("模型配置")
        model_layout = QGridLayout()
        model_layout.setSpacing(10)
        
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #374151;
                color: #f3f4f6;
                border: 1px solid #4b5563;
                border-radius: 6px;
                padding: 6px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        
        self.load_model_btn = ModernButton("加载模型", primary=True)
        self.model_status = QLabel("⚪ 未加载")
        self.model_status.setStyleSheet("color: #fbbf24;")
        
        model_layout.addWidget(QLabel("权重文件:"), 0, 0)
        model_layout.addWidget(self.model_combo, 0, 1)
        model_layout.addWidget(self.load_model_btn, 1, 0, 1, 2)
        model_layout.addWidget(QLabel("状态:"), 2, 0)
        model_layout.addWidget(self.model_status, 2, 1)
        
        self.model_group.setLayout(model_layout)
        layout.addWidget(self.model_group)
        
        # 输入源
        self.input_group = ModernGroupBox("输入源")
        input_layout = QGridLayout()
        input_layout.setSpacing(10)
        
        self.img_btn = ModernButton("📷 选择图片")
        self.video_btn = ModernButton("🎬 选择视频")
        self.camera_btn = ModernButton("📹 开启摄像头", primary=True)
        
        input_layout.addWidget(self.img_btn, 0, 0)
        input_layout.addWidget(self.video_btn, 0, 1)
        input_layout.addWidget(self.camera_btn, 1, 0, 1, 2)
        
        self.input_group.setLayout(input_layout)
        layout.addWidget(self.input_group)
        
        # 检测参数
        self.param_group = ModernGroupBox("检测参数")
        param_layout = QGridLayout()
        param_layout.setSpacing(10)
        
        # 置信度
        param_layout.addWidget(QLabel("置信度:"), 0, 0)
        self.conf_slider = ModernSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("0.25")
        self.conf_label.setMinimumWidth(40)
        param_layout.addWidget(self.conf_slider, 0, 1)
        param_layout.addWidget(self.conf_label, 0, 2)
        
        # IOU
        param_layout.addWidget(QLabel("IOU:"), 1, 0)
        self.iou_slider = ModernSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.iou_label = QLabel("0.45")
        param_layout.addWidget(self.iou_slider, 1, 1)
        param_layout.addWidget(self.iou_label, 1, 2)
        
        # 最大检测数
        param_layout.addWidget(QLabel("最大检测:"), 2, 0)
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 300)
        self.max_det_spin.setValue(300)
        self.max_det_spin.setStyleSheet("""
            QSpinBox {
                background-color: #374151;
                color: #f3f4f6;
                border: 1px solid #4b5563;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        param_layout.addWidget(self.max_det_spin, 2, 1, 1, 2)
        
        # 设备
        param_layout.addWidget(QLabel("设备:"), 3, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu"])
        self.device_combo.setStyleSheet("""
            QComboBox {
                background-color: #374151;
                color: #f3f4f6;
                border: 1px solid #4b5563;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        param_layout.addWidget(self.device_combo, 3, 1, 1, 2)
        
        # 显示选项
        self.show_labels_cb = QCheckBox("显示标签")
        self.show_conf_cb = QCheckBox("显示置信度")
        self.show_labels_cb.setChecked(True)
        self.show_conf_cb.setChecked(True)
        self.show_labels_cb.setStyleSheet("color: #f3f4f6;")
        self.show_conf_cb.setStyleSheet("color: #f3f4f6;")
        param_layout.addWidget(self.show_labels_cb, 4, 0, 1, 2)
        param_layout.addWidget(self.show_conf_cb, 4, 2)
        
        self.param_group.setLayout(param_layout)
        layout.addWidget(self.param_group)
        
        # 控制按钮
        self.control_group = ModernGroupBox("播放控制")
        control_layout = QHBoxLayout()
        
        self.pause_btn = ModernButton("⏸ 暂停")
        self.pause_btn.setEnabled(False)
        self.stop_btn = ModernButton("⏹ 停止", danger=True)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.stop_btn)
        
        self.control_group.setLayout(control_layout)
        layout.addWidget(self.control_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #374151;
                border-radius: 6px;
                text-align: center;
                color: #f3f4f6;
                background-color: #1f2937;
            }
            QProgressBar::chunk {
                background-color: #6366f1;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 结果显示区域
        self.result_group = ModernGroupBox("检测结果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(180)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                font-family: monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        self.result_text.setPlaceholderText("检测结果将显示在这里...")
        result_layout.addWidget(self.result_text)
        self.result_group.setLayout(result_layout)
        layout.addWidget(self.result_group)
        
        # 状态栏
        self.status_label = QLabel("✅ 就绪")
        self.status_label.setStyleSheet("""
            color: #10b981;
            padding: 8px;
            background-color: #1f2937;
            border-radius: 6px;
        """)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # 连接信号
        self.setup_connections()
        
    def setup_connections(self):
        self.load_model_btn.clicked.connect(lambda: self.load_model_clicked.emit(self.model_combo.currentText()))
        self.img_btn.clicked.connect(self.load_image_clicked)
        self.video_btn.clicked.connect(self.load_video_clicked)
        self.camera_btn.clicked.connect(self.toggle_camera_clicked)
        self.pause_btn.clicked.connect(self.pause_clicked)
        self.stop_btn.clicked.connect(self.stop_clicked)
    
        self.conf_slider.valueChanged.connect(self.update_conf_label)  # 新增
        self.conf_slider.valueChanged.connect(self._on_params_changed)
        self.iou_slider.valueChanged.connect(self.update_iou_label)    # 新增
        self.iou_slider.valueChanged.connect(self._on_params_changed)
        self.max_det_spin.valueChanged.connect(self._on_params_changed)
        self.show_labels_cb.stateChanged.connect(self._on_params_changed)
        self.show_conf_cb.stateChanged.connect(self._on_params_changed)

    def update_conf_label(self, value):
        conf = value / 100.0
        self.conf_label.setText(f"{conf:.2f}")

    def update_iou_label(self, value):
        """更新IOU标签显示"""
        iou = value / 100.0
        self.iou_label.setText(f"{iou:.2f}")
        
    def _on_params_changed(self):
        """参数变化时发送信号"""
        params = self.get_params()
        self.params_changed.emit(params)
        
    def get_params(self):
        """获取当前检测参数"""
        return {
            'conf': self.conf_slider.value() / 100.0,
            'iou': self.iou_slider.value() / 100.0,
            'max_det': self.max_det_spin.value(),
            'show_labels': self.show_labels_cb.isChecked(),
            'show_conf': self.show_conf_cb.isChecked(),
            'device': self.device_combo.currentText()
        }
        
    def update_progress(self, value, message=None):
        """更新进度条"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
        if value >= 100:
            self.progress_bar.setVisible(False)
            
    def set_model_status(self, loaded, message):
        """设置模型状态"""
        if loaded:
            self.model_status.setText("✅ 已加载")
            self.model_status.setStyleSheet("color: #10b981;")
        else:
            self.model_status.setText("❌ 加载失败")
            self.model_status.setStyleSheet("color: #ef4444;")
        self.status_label.setText(message)
        
    def set_playing_controls(self, playing):
        """设置播放控制按钮状态"""
        self.pause_btn.setEnabled(playing)
        self.stop_btn.setEnabled(playing)
        self.img_btn.setEnabled(not playing)
        self.video_btn.setEnabled(not playing)
        self.camera_btn.setEnabled(not playing)
        
    def set_pause_button(self, paused):
        """设置暂停按钮文字"""
        if paused:
            self.pause_btn.setText("▶ 继续")
        else:
            self.pause_btn.setText("⏸ 暂停")
            
    def set_camera_button(self, active):
        """设置摄像头按钮状态"""
        if active:
            self.camera_btn.setText("⏹ 关闭摄像头")
            self.camera_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                }
            """)
        else:
            self.camera_btn.setText("📹 开启摄像头")
            self.camera_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6366f1;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                }
            """)
            
    def update_result_text(self, detections):
        """更新检测结果文本"""
        if detections:
            total_count = len(detections)
            class_counts = {}
            for det in detections:
                label = det.get('label', 'unknown')
                class_counts[label] = class_counts.get(label, 0) + 1
            
            result_text = f"📊 检测统计 (共 {total_count} 个目标)\n"
            result_text += "=" * 40 + "\n\n"
            
            for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                result_text += f"  • {label}: {count} 个\n"
            
            result_text += "\n" + "=" * 40 + "\n"
            result_text += "📋 详细列表:\n\n"
            
            for i, det in enumerate(detections[:15]):
                label = det.get('label', 'unknown')
                conf = det.get('confidence', 0)
                result_text += f"  {i+1}. {label} (置信度: {conf:.2%})\n"
                
            if len(detections) > 15:
                result_text += f"\n  ... 还有 {len(detections) - 15} 个目标"
                
            self.result_text.setText(result_text)
        else:
            self.result_text.setText("未检测到任何目标")


class DisplayPanel(QWidget):
    """显示面板组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 分割器
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet("QSplitter::handle { background-color: #374151; }")
        
        # 图像显示区域
        self.image_label = ImageDisplayLabel(self)
        
        # 信息栏
        info_widget = QFrame()
        info_widget.setStyleSheet("""
            QFrame {
                background-color: #1f2937;
                border-radius: 12px;
                padding: 8px;
            }
        """)
        info_layout = QHBoxLayout(info_widget)
        
        self.fps_label = QLabel("📊 FPS: --")
        self.fps_label.setStyleSheet("color: #818cf8; font-size: 13px;")
        self.resolution_label = QLabel("📐 分辨率: --")
        self.resolution_label.setStyleSheet("color: #818cf8; font-size: 13px;")
        self.detection_count_label = QLabel("🎯 检测目标: 0")
        self.detection_count_label.setStyleSheet("color: #10b981; font-size: 13px;")
        self.inference_time_label = QLabel("⚡ 推理时间: --ms")
        self.inference_time_label.setStyleSheet("color: #fbbf24; font-size: 13px;")
        
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.resolution_label)
        info_layout.addWidget(self.detection_count_label)
        info_layout.addWidget(self.inference_time_label)
        info_layout.addStretch()
        
        splitter.addWidget(self.image_label)
        splitter.addWidget(info_widget)
        splitter.setSizes([550, 60])
        
        layout.addWidget(splitter)
        
    def display_image(self, img, detections=None, fps=None, inference_time=None):
        """显示图像"""
        if img is None:
            return
            
        # 转换图像格式
        if len(img.shape) == 3:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            height, width = img.shape
            q_img = QImage(img.data, width, height, QImage.Format_Grayscale8)
            
        # 缩放显示
        pixmap = QPixmap.fromImage(q_img)
        label_size = self.image_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        
        # 更新信息
        self.resolution_label.setText(f"📐 分辨率: {width} x {height}")
        
        if detections is not None:
            self.detection_count_label.setText(f"🎯 检测目标: {len(detections)}")
            
        if fps is not None:
            self.fps_label.setText(f"📊 FPS: {fps:.1f}")
            
        if inference_time is not None:
            self.inference_time_label.setText(f"⚡ 推理时间: {inference_time:.1f}ms")


def get_global_stylesheet():
    """获取全局样式表"""
    return """
        QMainWindow {
            background-color: #0f172a;
        }
        QScrollBar:vertical {
            background-color: #1e293b;
            width: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background-color: #475569;
            border-radius: 5px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #6366f1;
        }
        QToolTip {
            background-color: #1f2937;
            color: #f3f4f6;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 4px;
        }
    """