"""
主程序入口 - YOLOv11目标检测系统
支持单张检测、视频检测、摄像头检测、批量图片检测
所有检测结果自动保存到 runs 目录
"""

import sys
import os
import cv2
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QComboBox, QGroupBox, QGridLayout, QSlider, 
                             QSpinBox, QProgressBar, QMessageBox, QSplitter,
                             QTextEdit, QCheckBox, QFrame, QListWidget,
                             QListWidgetItem, QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from ultralytics import YOLO
from datetime import datetime
import time
import numpy as np


class InferenceThread(QThread):
    """单张图片推理线程 - 自动保存结果"""
    
    finished = pyqtSignal(object, list, float, str)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, model, image_path, conf, iou, max_det):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        
    def run(self):
        try:
            self.progress.emit(20)
            
            img = cv2.imread(self.image_path)
            if img is None:
                self.error.emit(f"无法读取图片: {self.image_path}")
                return
            
            self.progress.emit(40)
            start = time.time()
            results = self.model(img, conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False)
            elapsed = (time.time() - start) * 1000
            
            self.progress.emit(70)
            annotated = results[0].plot()
            annotated = annotated.copy()  # 创建可写副本
            
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    detections.append({
                        'label': results[0].names[int(box.cls[0])],
                        'confidence': float(box.conf[0])
                    })
            
            self.progress.emit(90)
            
            # 保存结果图片到 runs/image_results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.basename(self.image_path)
            name_without_ext = os.path.splitext(filename)[0]
            save_dir = os.path.join('runs', 'image_results')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{name_without_ext}_{timestamp}.jpg')
            cv2.imwrite(save_path, annotated)
            
            self.progress.emit(100)
            self.finished.emit(annotated, detections, elapsed, save_path)
            
        except Exception as e:
            self.error.emit(str(e))


class VideoThread(QThread):
    """视频处理线程 - 自动保存结果"""
    
    frame_ready = pyqtSignal(object, list, float, float, int, int)
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, model, video_path, conf, iou, max_det):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.running = True
        self.paused = False
        self.save_dir = None
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"无法打开视频: {self.video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_input = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建保存目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.save_dir = os.path.join('runs', 'video_results', f'{video_name}_{timestamp}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 输出视频路径
        output_path = os.path.join(self.save_dir, 'detection_result.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))
        
        frame_count = 0
        frame_times = []
        skip_frames = max(1, int(fps_input / 30))
        
        while self.running and frame_count < total_frames:
            while self.paused and self.running:
                self.msleep(100)
                
            ret, frame = cap.read()
            if not ret:
                break
                
            should_process = (frame_count % skip_frames == 0)
            
            if should_process:
                start = time.time()
                results = self.model(frame, conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False)
                inference_time = (time.time() - start) * 1000
                
                frame_times.append(inference_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                avg_time = sum(frame_times) / len(frame_times)
                current_fps = 1000 / avg_time if avg_time > 0 else 0
                
                annotated = results[0].plot()
                annotated = annotated.copy()  # 创建可写副本
                
                cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        detections.append({
                            'label': results[0].names[int(box.cls[0])],
                            'confidence': float(box.conf[0])
                        })
                
                writer.write(annotated)
                self.frame_ready.emit(annotated, detections, current_fps, inference_time, frame_count, total_frames)
            else:
                writer.write(frame)
                self.frame_ready.emit(frame, [], 0, 0, frame_count, total_frames)
            
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            self.progress.emit(progress)
            
        cap.release()
        writer.release()
        
        # 保存检测信息
        info_path = os.path.join(self.save_dir, 'detection_info.txt')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("YOLOv11 视频检测结果报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"源文件: {self.video_path}\n")
            f.write(f"总帧数: {frame_count}\n")
            f.write(f"输出视频: {output_path}\n")
            f.write("=" * 50 + "\n")
            
        self.finished.emit()
        
    def stop(self):
        self.running = False
        
    def pause(self):
        self.paused = True
        
    def resume(self):
        self.paused = False


class CameraThread(QThread):
    """摄像头处理线程"""
    
    frame_ready = pyqtSignal(object, list, float, float)
    error = pyqtSignal(str)
    
    def __init__(self, model, conf, iou, max_det):
        super().__init__()
        self.model = model
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.running = True
        self.cap = None
        
    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error.emit("无法打开摄像头")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_times = []
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            start = time.time()
            results = self.model(frame, conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False)
            inference_time = (time.time() - start) * 1000
            
            frame_times.append(inference_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_time = sum(frame_times) / len(frame_times)
            current_fps = 1000 / avg_time if avg_time > 0 else 0
            
            annotated = results[0].plot()
            annotated = annotated.copy()  # 创建可写副本
            
            cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    detections.append({
                        'label': results[0].names[int(box.cls[0])],
                        'confidence': float(box.conf[0])
                    })
                    
            self.frame_ready.emit(annotated, detections, current_fps, inference_time)
            
        if self.cap:
            self.cap.release()
            
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class BatchInferenceThread(QThread):
    """批量图片推理线程 - 自动保存结果"""
    
    progress = pyqtSignal(int, int, str)
    image_done = pyqtSignal(object, list, float, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, model, image_paths, conf, iou, max_det, save_results=True):
        super().__init__()
        self.model = model
        self.image_paths = image_paths
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.save_results = save_results
        self.running = True
        
    def run(self):
        total = len(self.image_paths)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join('runs', 'batch_results', f'batch_{timestamp}')
        
        if self.save_results:
            os.makedirs(save_dir, exist_ok=True)
        
        for i, img_path in enumerate(self.image_paths):
            if not self.running:
                break
                
            try:
                filename = os.path.basename(img_path)
                self.progress.emit(i + 1, total, filename)
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                start = time.time()
                results = self.model(img, conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False)
                elapsed = (time.time() - start) * 1000
                
                annotated = results[0].plot()
                annotated = annotated.copy()  # 创建可写副本
                
                if self.save_results:
                    name_without_ext = os.path.splitext(filename)[0]
                    save_path = os.path.join(save_dir, f'{name_without_ext}_result.jpg')
                    cv2.imwrite(save_path, annotated)
                
                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        detections.append({
                            'label': results[0].names[int(box.cls[0])],
                            'confidence': float(box.conf[0])
                        })
                
                self.image_done.emit(annotated, detections, elapsed, filename)
                
            except Exception as e:
                self.error.emit(f"处理 {filename} 时出错: {str(e)}")
                
        # 保存批量报告
        if self.save_results:
            report_path = os.path.join(save_dir, 'batch_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("YOLOv11 批量检测报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总图片数: {total}\n")
                f.write("=" * 50 + "\n")
                
        self.finished.emit()
        
    def stop(self):
        self.running = False


class YOLOv11UI(QMainWindow):
    """主界面 - 支持批量处理，所有结果自动保存"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_thread = None
        self.video_thread = None
        self.camera_thread = None
        self.batch_thread = None
        self.is_processing = False
        self.batch_image_paths = []
        self.batch_results = []
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("YOLOv11 校园垃圾检测系统 - 支持批量处理")
        self.setGeometry(100, 100, 1600, 950)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #45475a; }
            QPushButton:disabled { background-color: #1e1e2e; color: #6c7086; }
            QPushButton#primary_btn {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QPushButton#primary_btn:hover { background-color: #b4befe; }
            QPushButton#danger_btn {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
            QGroupBox {
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 8px;
                margin-top: 12px;
                font-size: 13px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #cdd6f4; }
            QListWidget {
                background-color: #11111b;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 6px;
                font-size: 11px;
            }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background-color: #89b4fa; color: #1e1e2e; }
            QTabWidget::pane {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #cdd6f4;
                padding: 6px 12px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected { background-color: #89b4fa; color: #1e1e2e; }
            QComboBox, QSpinBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
            }
            QTextEdit {
                background-color: #11111b;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 6px;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #313244;
                border-radius: 4px;
                text-align: center;
                color: #cdd6f4;
            }
            QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
            QSlider::groove:horizontal {
                height: 4px;
                background: #313244;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QScrollArea { border: none; background-color: transparent; }
        """)
        
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # ==================== 左侧控制面板 ====================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel("YOLOv11 校园垃圾检测系统")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa; padding: 5px;")
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)
        
        self.tab_widget = QTabWidget()
        
        # ========== 标签页1: 单张检测 ==========
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)
        
        model_group = QGroupBox("模型配置")
        model_layout = QGridLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.load_btn = QPushButton("加载模型")
        self.load_btn.setObjectName("primary_btn")
        self.model_status = QLabel("⚪ 未加载")
        self.model_status.setStyleSheet("color: #f9e2af;")
        model_layout.addWidget(QLabel("权重:"), 0, 0)
        model_layout.addWidget(self.model_combo, 0, 1)
        model_layout.addWidget(self.load_btn, 1, 0, 1, 2)
        model_layout.addWidget(QLabel("状态:"), 2, 0)
        model_layout.addWidget(self.model_status, 2, 1)
        model_group.setLayout(model_layout)
        single_layout.addWidget(model_group)
        
        input_group = QGroupBox("输入源")
        input_layout = QGridLayout()
        self.img_btn = QPushButton("📷 选择图片")
        self.video_btn = QPushButton("🎬 选择视频")
        self.camera_btn = QPushButton("📹 开启摄像头")
        input_layout.addWidget(self.img_btn, 0, 0)
        input_layout.addWidget(self.video_btn, 0, 1)
        input_layout.addWidget(self.camera_btn, 1, 0, 1, 2)
        input_group.setLayout(input_layout)
        single_layout.addWidget(input_group)
        
        param_group = QGroupBox("检测参数")
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("置信度:"), 0, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("0.25")
        param_layout.addWidget(self.conf_slider, 0, 1)
        param_layout.addWidget(self.conf_label, 0, 2)
        
        param_layout.addWidget(QLabel("IOU:"), 1, 0)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.iou_label = QLabel("0.45")
        param_layout.addWidget(self.iou_slider, 1, 1)
        param_layout.addWidget(self.iou_label, 1, 2)
        
        param_layout.addWidget(QLabel("最大检测:"), 2, 0)
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 300)
        self.max_det_spin.setValue(300)
        param_layout.addWidget(self.max_det_spin, 2, 1, 1, 2)
        
        self.show_labels_cb = QCheckBox("显示标签")
        self.show_conf_cb = QCheckBox("显示置信度")
        self.show_labels_cb.setChecked(True)
        self.show_conf_cb.setChecked(True)
        param_layout.addWidget(self.show_labels_cb, 3, 0, 1, 2)
        param_layout.addWidget(self.show_conf_cb, 3, 2)
        
        param_group.setLayout(param_layout)
        single_layout.addWidget(param_group)
        
        single_layout.addStretch()
        
        # ========== 标签页2: 批量检测 ==========
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
        
        batch_control_group = QGroupBox("批量处理控制")
        batch_control_layout = QHBoxLayout()
        
        self.select_images_btn = QPushButton("📁 选择多张图片")
        self.select_folder_btn = QPushButton("📂 选择文件夹")
        self.clear_list_btn = QPushButton("🗑️ 清空列表")
        self.start_batch_btn = QPushButton("🚀 开始批量检测")
        self.start_batch_btn.setObjectName("primary_btn")
        self.start_batch_btn.setEnabled(False)
        self.stop_batch_btn = QPushButton("⏹️ 停止检测")
        self.stop_batch_btn.setObjectName("danger_btn")
        self.stop_batch_btn.setEnabled(False)
        
        batch_control_layout.addWidget(self.select_images_btn)
        batch_control_layout.addWidget(self.select_folder_btn)
        batch_control_layout.addWidget(self.clear_list_btn)
        batch_control_layout.addWidget(self.start_batch_btn)
        batch_control_layout.addWidget(self.stop_batch_btn)
        batch_control_group.setLayout(batch_control_layout)
        batch_layout.addWidget(batch_control_group)
        
        list_group = QGroupBox("待处理图片列表")
        list_layout = QVBoxLayout()
        
        self.image_list_widget = QListWidget()
        self.image_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.image_list_widget.setMinimumHeight(150)
        
        list_info_layout = QHBoxLayout()
        self.list_count_label = QLabel("已选择: 0 张图片")
        self.list_count_label.setStyleSheet("color: #a6e3a1;")
        self.remove_selected_btn = QPushButton("移除选中")
        self.remove_selected_btn.setEnabled(False)
        list_info_layout.addWidget(self.list_count_label)
        list_info_layout.addStretch()
        list_info_layout.addWidget(self.remove_selected_btn)
        
        list_layout.addWidget(self.image_list_widget)
        list_layout.addLayout(list_info_layout)
        list_group.setLayout(list_layout)
        batch_layout.addWidget(list_group)
        
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        batch_layout.addWidget(self.batch_progress_bar)
        
        batch_result_group = QGroupBox("批量检测统计")
        batch_result_layout = QVBoxLayout()
        self.batch_result_text = QTextEdit()
        self.batch_result_text.setReadOnly(True)
        self.batch_result_text.setMaximumHeight(150)
        batch_result_layout.addWidget(self.batch_result_text)
        batch_result_group.setLayout(batch_result_layout)
        batch_layout.addWidget(batch_result_group)
        
        batch_layout.addStretch()
        
        # ========== 标签页3: 结果历史 ==========
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        history_control_layout = QHBoxLayout()
        self.refresh_history_btn = QPushButton("🔄 刷新历史")
        self.clear_history_btn = QPushButton("🗑️ 清空历史")
        history_control_layout.addWidget(self.refresh_history_btn)
        history_control_layout.addWidget(self.clear_history_btn)
        history_control_layout.addStretch()
        history_layout.addLayout(history_control_layout)
        
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.on_history_double_click)
        history_layout.addWidget(self.history_list)
        
        self.tab_widget.addTab(single_tab, "单张检测")
        self.tab_widget.addTab(batch_tab, "批量检测")
        self.tab_widget.addTab(history_tab, "检测历史")
        
        left_layout.addWidget(self.tab_widget)
        
        control_group = QGroupBox("播放控制")
        control_layout = QHBoxLayout()
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.stop_btn)
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(180)
        self.result_text.setMaximumHeight(250)
        self.result_text.setPlaceholderText("检测结果将显示在这里...")
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        left_layout.addWidget(result_group)
        
        self.status_label = QLabel("✅ 就绪 - 请加载模型")
        self.status_label.setStyleSheet("padding: 8px; background-color: #313244; border-radius: 6px; font-size: 11px;")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # ==================== 右侧显示区域 ====================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(5)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: #11111b; 
            border-radius: 8px;
            border: 1px solid #313244;
        """)
        self.image_label.setMinimumHeight(550)
        self.image_label.setText("等待图像加载...")
        
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 6px;
                padding: 6px;
            }
            QLabel {
                font-size: 13px;
                font-weight: bold;
            }
        """)
        info_layout = QHBoxLayout(info_frame)
        info_layout.setSpacing(20)
        info_layout.setContentsMargins(15, 5, 15, 5)
        
        self.fps_label = QLabel("📊 FPS: --")
        self.fps_label.setStyleSheet("color: #89b4fa; font-size: 13px;")
        self.res_label = QLabel("📐 分辨率: --")
        self.res_label.setStyleSheet("color: #89b4fa; font-size: 13px;")
        self.count_label = QLabel("🎯 检测目标: 0")
        self.count_label.setStyleSheet("color: #a6e3a1; font-size: 13px;")
        self.time_label = QLabel("⚡ 推理时间: --ms")
        self.time_label.setStyleSheet("color: #f9e2af; font-size: 13px;")
        
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.res_label)
        info_layout.addWidget(self.count_label)
        info_layout.addWidget(self.time_label)
        info_layout.addStretch()
        
        right_layout.addWidget(self.image_label, 1)
        right_layout.addWidget(info_frame, 0)
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([450, 1150])
        
        self.setup_connections()
        self.scan_models()
        self.refresh_history_list()
        
    def setup_connections(self):
        """设置信号连接"""
        self.load_btn.clicked.connect(self.load_model)
        self.img_btn.clicked.connect(self.load_image)
        self.video_btn.clicked.connect(self.load_video)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.pause_btn.clicked.connect(self.pause_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        self.select_images_btn.clicked.connect(self.select_images)
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.clear_list_btn.clicked.connect(self.clear_image_list)
        self.start_batch_btn.clicked.connect(self.start_batch_detection)
        self.stop_batch_btn.clicked.connect(self.stop_batch_detection)
        self.remove_selected_btn.clicked.connect(self.remove_selected_images)
        self.image_list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        
        self.refresh_history_btn.clicked.connect(self.refresh_history_list)
        self.clear_history_btn.clicked.connect(self.clear_history)
        
        self.conf_slider.valueChanged.connect(self.update_conf_display)
        self.iou_slider.valueChanged.connect(self.update_iou_display)
        
    def scan_models(self):
        """扫描模型文件"""
        if os.path.exists("weight"):
            for f in os.listdir("weight"):
                if f.endswith(('.pt', '.pth')):
                    self.model_combo.addItem(os.path.join("weight", f))
        self.model_combo.addItem("yolo11n.pt")
        self.model_combo.addItem("yolo11s.pt")
        
    def update_conf_display(self, val):
        self.conf_label.setText(f"{val/100:.2f}")
        
    def update_iou_display(self, val):
        self.iou_label.setText(f"{val/100:.2f}")
        
    def load_model(self):
        """加载模型"""
        weight = self.model_combo.currentText()
        if not os.path.exists(weight):
            QMessageBox.warning(self, "警告", f"模型不存在: {weight}")
            return
            
        self.status_label.setText("🔄 正在加载模型...")
        self.load_btn.setEnabled(False)
        
        try:
            self.model = YOLO(weight)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model_status.setText(f"✅ 已加载 ({device})")
            self.model_status.setStyleSheet("color: #a6e3a1;")
            self.status_label.setText(f"✅ 模型加载成功 - 设备: {device}")
            self.start_batch_btn.setEnabled(len(self.batch_image_paths) > 0)
        except Exception as e:
            self.model_status.setText("❌ 加载失败")
            self.model_status.setStyleSheet("color: #f38ba8;")
            QMessageBox.critical(self, "错误", str(e))
        finally:
            self.load_btn.setEnabled(True)
            
    def clear_results(self):
        """清空所有结果显示"""
        self.result_text.clear()
        self.result_text.setPlaceholderText("检测结果将显示在这里...")
        self.count_label.setText("🎯 检测目标: 0")
        self.image_label.clear()
        self.image_label.setText("等待图像加载...")
        self.fps_label.setText("📊 FPS: --")
        self.time_label.setText("⚡ 推理时间: --ms")
        self.res_label.setText("📐 分辨率: --")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
    # ==================== 单张检测 ====================
    def load_image(self):
        """加载并检测图片"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        self.stop_processing()
        self.clear_results()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "data/images", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.process_image(file_path)
            
    def process_image(self, file_path):
        """处理单张图片"""
        self.is_processing = True
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"📷 正在处理: {os.path.basename(file_path)}")
        
        self.current_thread = InferenceThread(
            self.model, file_path,
            self.conf_slider.value() / 100,
            self.iou_slider.value() / 100,
            self.max_det_spin.value()
        )
        self.current_thread.progress.connect(self.progress_bar.setValue)
        self.current_thread.finished.connect(self.on_image_done)
        self.current_thread.error.connect(self.on_error)
        self.current_thread.start()
        
    def on_image_done(self, image, detections, elapsed, save_path):
        """单张图片处理完成"""
        self.display_image(image, detections)
        self.time_label.setText(f"⚡ 推理时间: {elapsed:.1f}ms")
        self.update_result_text(detections, "单张检测")
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"✅ 检测完成，结果已保存至: {save_path}")
        self.is_processing = False
        self.save_to_history(detections, elapsed)
        
    # ==================== 批量检测 ====================
    def select_images(self):
        """选择多张图片"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择多张图片", "data/images", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        for f in files:
            if f not in self.batch_image_paths:
                self.batch_image_paths.append(f)
                item = QListWidgetItem(f"📷 {os.path.basename(f)}")
                item.setToolTip(f)
                self.image_list_widget.addItem(item)
        
        self.update_list_count()
        self.start_batch_btn.setEnabled(len(self.batch_image_paths) > 0 and self.model is not None)
        
    def select_folder(self):
        """选择文件夹中的所有图片"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "data/images")
        
        if folder:
            extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            files = []
            for f in os.listdir(folder):
                if f.lower().endswith(extensions):
                    files.append(os.path.join(folder, f))
            
            for f in files:
                if f not in self.batch_image_paths:
                    self.batch_image_paths.append(f)
                    item = QListWidgetItem(f"📁 {os.path.basename(f)}")
                    item.setToolTip(f)
                    self.image_list_widget.addItem(item)
            
            self.update_list_count()
            self.start_batch_btn.setEnabled(len(self.batch_image_paths) > 0 and self.model is not None)
            self.status_label.setText(f"✅ 已添加 {len(files)} 张图片")
            
    def clear_image_list(self):
        """清空图片列表"""
        self.batch_image_paths.clear()
        self.image_list_widget.clear()
        self.update_list_count()
        self.start_batch_btn.setEnabled(False)
        self.batch_result_text.clear()
        
    def remove_selected_images(self):
        """移除选中的图片"""
        selected_items = self.image_list_widget.selectedItems()
        for item in selected_items:
            idx = self.image_list_widget.row(item)
            self.image_list_widget.takeItem(idx)
            if idx < len(self.batch_image_paths):
                del self.batch_image_paths[idx]
        self.update_list_count()
        self.start_batch_btn.setEnabled(len(self.batch_image_paths) > 0 and self.model is not None)
        
    def on_selection_changed(self):
        """列表选择变化"""
        self.remove_selected_btn.setEnabled(len(self.image_list_widget.selectedItems()) > 0)
        
    def update_list_count(self):
        """更新列表计数"""
        self.list_count_label.setText(f"已选择: {len(self.batch_image_paths)} 张图片")
        
    def start_batch_detection(self):
        """开始批量检测"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        if len(self.batch_image_paths) == 0:
            QMessageBox.warning(self, "警告", "请先选择图片")
            return
            
        print(f"开始批量检测，共 {len(self.batch_image_paths)} 张图片")  # 调试信息
            
        self.is_processing = True
        self.start_batch_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        self.select_images_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)
        self.clear_list_btn.setEnabled(False)
        
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setValue(0)
        self.batch_result_text.clear()
        self.batch_results = []
        
        self.status_label.setText("🚀 开始批量检测...")
        
        self.batch_thread = BatchInferenceThread(
            self.model,
            self.batch_image_paths.copy(),  # 使用副本避免原列表被修改
            self.conf_slider.value() / 100,
            self.iou_slider.value() / 100,
            self.max_det_spin.value(),
            save_results=True
        )
        self.batch_thread.progress.connect(self.on_batch_progress)
        self.batch_thread.image_done.connect(self.on_batch_image_done)
        self.batch_thread.finished.connect(self.on_batch_finished)
        self.batch_thread.error.connect(self.on_error)
        self.batch_thread.start()
        
    def on_batch_progress(self, current, total, filename):
        """批量检测进度更新"""
        self.batch_progress_bar.setMaximum(total)
        self.batch_progress_bar.setValue(current)
        self.status_label.setText(f"📷 正在处理: {filename} ({current}/{total})")
        
    def on_batch_image_done(self, image, detections, elapsed, filename):
        """单张批量图片处理完成"""
        self.batch_results.append({
            'filename': filename,
            'detections': detections,
            'time': elapsed
        })
        
        # 在右侧显示当前处理的图片
        self.display_image(image, detections)
        
        # 在结果文本中显示
        if detections:
            class_counts = {}
            for d in detections:
                label = d['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            
            result_line = f"📷 {filename}: {len(detections)}个目标"
            if class_counts:
                labels_str = ', '.join([f"{k}({v})" for k, v in class_counts.items()])
                result_line += f" [{labels_str}]"
            result_line += f" ⏱️ {elapsed:.0f}ms\n"
            self.batch_result_text.append(result_line)
        else:
            self.batch_result_text.append(f"📷 {filename}: 未检测到目标 ⏱️ {elapsed:.0f}ms\n")
        
        # 滚动到底部
        self.batch_result_text.verticalScrollBar().setValue(
            self.batch_result_text.verticalScrollBar().maximum()
        )
        
        self.save_to_history(detections, elapsed, filename)
        
    def on_batch_finished(self):
        """批量检测完成"""
        self.is_processing = False
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.select_images_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.clear_list_btn.setEnabled(True)
        self.batch_progress_bar.setVisible(False)
        
        total_images = len(self.batch_results)
        total_detections = sum(len(r['detections']) for r in self.batch_results)
        avg_time = sum(r['time'] for r in self.batch_results) / total_images if total_images > 0 else 0
        
        summary = f"\n{'='*50}\n✅ 批量检测完成!\n"
        summary += f"📊 总图片数: {total_images}\n"
        summary += f"🎯 总检测目标: {total_detections}\n"
        summary += f"⏱️ 平均耗时: {avg_time:.1f}ms\n"
        summary += f"💾 结果已保存至 runs/batch_results/ 目录\n"
        self.batch_result_text.append(summary)
        
        self.status_label.setText(f"✅ 批量检测完成! 共处理 {total_images} 张图片，检测到 {total_detections} 个目标")
        self.refresh_history_list()
        
    def stop_batch_detection(self):
        """停止批量检测"""
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.stop()
            self.batch_thread.wait()
        self.on_batch_finished()
        self.status_label.setText("⏹️ 批量检测已停止")
        
    # ==================== 视频/摄像头检测 ====================
    def load_video(self):
        """加载视频"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        self.stop_processing()
        self.clear_results()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "data/videos", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.start_video(file_path)
            
    def start_video(self, file_path):
        """开始视频检测"""
        self.is_processing = True
        self.progress_bar.setVisible(True)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.img_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.camera_btn.setEnabled(False)
        
        self.result_text.clear()
        self.result_text.setPlaceholderText("视频检测中，实时结果将显示在这里...")
        
        self.video_thread = VideoThread(
            self.model, file_path,
            self.conf_slider.value() / 100,
            self.iou_slider.value() / 100,
            self.max_det_spin.value()
        )
        self.video_thread.frame_ready.connect(self.on_video_frame)
        self.video_thread.progress.connect(self.progress_bar.setValue)
        self.video_thread.finished.connect(self.on_video_finished)
        self.video_thread.error.connect(self.on_error)
        self.video_thread.start()
        
    def on_video_frame(self, frame, detections, fps, inference_time, current, total):
        """视频帧处理"""
        self.display_image(frame, detections)
        self.fps_label.setText(f"📊 FPS: {fps:.1f}")
        self.time_label.setText(f"⚡ 推理时间: {inference_time:.1f}ms")
        self.status_label.setText(f"🎬 处理中: {current}/{total}")
        
        if detections:
            self.update_result_text(detections, f"实时检测 (帧 {current}/{total})")
            
    def on_video_finished(self):
        """视频处理完成"""
        self.reset_ui()
        self.status_label.setText("✅ 视频处理完成，结果已保存至 runs/video_results/ 目录")
        
    def toggle_camera(self):
        """切换摄像头"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        else:
            self.start_camera()
            
    def start_camera(self):
        """启动摄像头"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        self.stop_processing()
        self.clear_results()
        
        self.is_processing = True
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.img_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.camera_btn.setText("关闭摄像头")
        
        self.result_text.clear()
        self.result_text.setPlaceholderText("摄像头实时检测中...")
        
        self.camera_thread = CameraThread(
            self.model,
            self.conf_slider.value() / 100,
            self.iou_slider.value() / 100,
            self.max_det_spin.value()
        )
        self.camera_thread.frame_ready.connect(self.on_camera_frame)
        self.camera_thread.error.connect(self.on_error)
        self.camera_thread.start()
        
    def on_camera_frame(self, frame, detections, fps, inference_time):
        """摄像头帧处理"""
        self.display_image(frame, detections)
        self.fps_label.setText(f"📊 FPS: {fps:.1f}")
        self.time_label.setText(f"⚡ 推理时间: {inference_time:.1f}ms")
        
        if detections:
            self.update_result_text(detections, "摄像头实时检测")
            
    def stop_camera(self):
        """停止摄像头"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        self.camera_btn.setText("📹 开启摄像头")
        self.reset_ui()
        self.status_label.setText("✅ 摄像头已关闭")
        
    # ==================== 通用方法 ====================
    def pause_processing(self):
        """暂停/继续处理"""
        if self.video_thread:
            if self.pause_btn.text() == "⏸ 暂停":
                self.video_thread.pause()
                self.pause_btn.setText("▶ 继续")
                self.status_label.setText("⏸ 已暂停")
            else:
                self.video_thread.resume()
                self.pause_btn.setText("⏸ 暂停")
                self.status_label.setText("🎬 处理中...")
                
    def stop_processing(self):
        """停止所有处理"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
            
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
            self.camera_btn.setText("📹 开启摄像头")
            
        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.terminate()
            self.current_thread.wait()
            self.current_thread = None
            
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.stop()
            self.batch_thread.wait()
            self.batch_thread = None
            
        self.reset_ui()
        
    def reset_ui(self):
        """重置UI状态"""
        self.is_processing = False
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self.img_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.camera_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
    def display_image(self, img, detections):
        """显示图像"""
        if img is None:
            return
            
        h, w = img.shape[:2]
        bytes_per_line = 3 * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        pixmap = QPixmap.fromImage(q_img)
        label_size = self.image_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
        
        self.res_label.setText(f"📐 分辨率: {w} x {h}")
        self.count_label.setText(f"🎯 检测目标: {len(detections)}")
        
    def update_result_text(self, detections, title="检测结果"):
        """更新检测结果文本"""
        if detections:
            class_counts = {}
            for d in detections:
                label = d['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            
            text = f"📊 {title} (共 {len(detections)} 个目标)\n"
            text += "=" * 40 + "\n\n"
            
            for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                text += f"  • {label}: {count} 个\n"
            
            text += "\n" + "=" * 40 + "\n"
            text += "📋 详细列表:\n\n"
            
            for i, d in enumerate(detections[:15]):
                text += f"  {i+1}. {d['label']} (置信度: {d['confidence']:.2%})\n"
                
            if len(detections) > 15:
                text += f"\n  ... 还有 {len(detections) - 15} 个目标"
                
            self.result_text.setText(text)
        else:
            self.result_text.setText("未检测到任何目标")
            
    def save_to_history(self, detections, elapsed, filename=""):
        """保存检测结果到历史"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if filename:
            title = filename
        else:
            title = "单张检测"
        
        if detections:
            class_counts = {}
            for d in detections:
                label = d['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            
            summary = f"{timestamp} - {title}: {len(detections)}个目标"
            if class_counts:
                labels_str = ', '.join([f"{k}({v})" for k, v in class_counts.items()])
                summary += f" [{labels_str}]"
            summary += f" ⏱️ {elapsed:.0f}ms"
        else:
            summary = f"{timestamp} - {title}: 无目标 ⏱️ {elapsed:.0f}ms"
        
        history_file = 'runs/detection_history.txt'
        os.makedirs('runs', exist_ok=True)
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(summary + '\n')
            
    def refresh_history_list(self):
        """刷新历史列表"""
        self.history_list.clear()
        history_file = 'runs/detection_history.txt'
        
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    item = QListWidgetItem(line.strip())
                    self.history_list.addItem(item)
                    
        if self.history_list.count() == 0:
            self.history_list.addItem("暂无检测历史记录")
            
    def on_history_double_click(self, item):
        pass
        
    def clear_history(self):
        """清空历史记录"""
        history_file = 'runs/detection_history.txt'
        if os.path.exists(history_file):
            os.remove(history_file)
        self.refresh_history_list()
        self.status_label.setText("✅ 历史记录已清空")
        
    def on_error(self, msg):
        """错误处理"""
        QMessageBox.critical(self, "错误", msg)
        self.stop_processing()
        self.status_label.setText(f"❌ 错误: {msg}")
        
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_processing()
        if self.batch_thread:
            self.batch_thread.stop()
            self.batch_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = YOLOv11UI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()