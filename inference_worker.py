"""
推理工作线程模块 - YOLOv11目标检测系统
"""

import os
import cv2
import torch
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from ultralytics import YOLO
from datetime import datetime
import time


class InferenceWorker(QThread):
    """YOLOv11推理工作线程"""
    
    model_loaded = pyqtSignal(bool, str)
    progress_update = pyqtSignal(int, str)
    result_ready = pyqtSignal(object, list, str)
    video_frame_ready = pyqtSignal(object, list, float, float)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, weight_path, device='auto'):
        super().__init__()
        self.weight_path = weight_path
        self.device = device
        self.model = None
        self.model_loaded_flag = False
        
        self.running = True
        self.paused = False
        self.mutex = QMutex()
        self.cond = QWaitCondition()
        
        self.current_task = None
        self.current_params = {
            'conf': 0.25,
            'iou': 0.45,
            'max_det': 300,
            'show_labels': True,
            'show_conf': True
        }
        
        self.frame_times = []
        self.fps = 0
        
        # 视频保存相关
        self.video_writer = None
        self.video_save_dir = None
        self.current_video_path = None
        
    def run(self):
        self.load_model()
        
        while self.running:
            self.mutex.lock()
            if self.paused:
                self.cond.wait(self.mutex)
            self.mutex.unlock()
            
            if self.current_task and self.model_loaded_flag:
                task_type = self.current_task.get('type')
                
                if task_type == 'image':
                    self.process_image_task()
                elif task_type == 'video':
                    self.process_video_task()
                elif task_type == 'video_frame':
                    self.process_single_frame_task()
                    
            time.sleep(0.01)
            
    def load_model(self):
        try:
            self.progress_update.emit(10, "正在加载模型权重...")
            self.model = YOLO(self.weight_path)
            
            self.progress_update.emit(50, "正在配置设备...")
            
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            if self.device != 'cpu' and torch.cuda.is_available():
                try:
                    _ = torch.zeros(1).cuda()
                except:
                    self.device = 'cpu'
            
            self.progress_update.emit(90, "模型准备就绪...")
            
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, verbose=False)
            
            self.model_loaded_flag = True
            self.progress_update.emit(100, "模型加载完成")
            self.model_loaded.emit(True, f"模型加载成功 (设备: {self.device})")
            
        except Exception as e:
            self.model_loaded_flag = False
            self.model_loaded.emit(False, f"模型加载失败: {str(e)}")
            self.error_occurred.emit(f"模型加载失败: {str(e)}")
            
    def update_params(self, **kwargs):
        """更新检测参数"""
        valid_params = ['conf', 'iou', 'max_det', 'show_labels', 'show_conf']
        for key, value in kwargs.items():
            if key in valid_params and key in self.current_params:
                self.current_params[key] = value
                
    def process_image(self, image_path, params):
        if not self.model_loaded_flag:
            self.error_occurred.emit("模型未加载")
            return
        self.current_task = {'type': 'image', 'path': image_path, 'params': params}
        
    def process_video(self, video_path, params):
        if not self.model_loaded_flag:
            self.error_occurred.emit("模型未加载")
            return
        self.current_task = {'type': 'video', 'path': video_path, 'params': params}
        
    def process_video_frame(self, frame, params):
        if not self.model_loaded_flag:
            return
        self.current_task = {'type': 'video_frame', 'frame': frame.copy(), 'params': params}
        
    def process_image_task(self):
        task = self.current_task
        self.current_task = None
        
        try:
            if task.get('params'):
                self.update_params(**task['params'])
                
            img = cv2.imread(task['path'])
            if img is None:
                self.error_occurred.emit(f"无法读取图片: {task['path']}")
                self.finished.emit()
                return
                
            self.progress_update.emit(30, "正在进行目标检测...")
            
            start_time = time.time()
            results = self.model(
                img, 
                conf=self.current_params['conf'],
                iou=self.current_params['iou'],
                max_det=self.current_params['max_det'],
                device=self.device,
                verbose=False
            )
            inference_time = (time.time() - start_time) * 1000
            
            detections = self.parse_detections(results[0])
            
            self.progress_update.emit(70, "正在绘制检测框...")
            
            # 绘制结果（返回可写的副本）
            annotated_img = self.draw_detections(
                img, results[0],
                self.current_params['show_labels'],
                self.current_params['show_conf']
            )
            
            # 添加推理时间文字
            cv2.putText(annotated_img, f"Inference: {inference_time:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 保存结果到 runs/image_results
            save_path = self.save_result(annotated_img, 'image', task['path'])
            
            self.progress_update.emit(100, "检测完成")
            self.result_ready.emit(annotated_img, detections, save_path)
            
        except Exception as e:
            self.error_occurred.emit(f"图片检测失败: {str(e)}")
        finally:
            self.finished.emit()
            
    def process_video_task(self):
        task = self.current_task
        self.current_task = None
        
        cap = None
        writer = None
        self.video_writer = None
        
        try:
            if task.get('params'):
                self.update_params(**task['params'])
                
            cap = cv2.VideoCapture(task['path'])
            if not cap.isOpened():
                self.error_occurred.emit(f"无法打开视频: {task['path']}")
                self.finished.emit()
                return
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_input = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 创建保存目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_name = os.path.splitext(os.path.basename(task['path']))[0]
            save_dir = os.path.join('runs', 'video_results', f'{video_name}_{timestamp}')
            os.makedirs(save_dir, exist_ok=True)
            self.video_save_dir = save_dir
            
            # 输出视频路径
            output_path = os.path.join(save_dir, 'detection_result.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))
            
            frame_count = 0
            processed_count = 0
            skip_frames = max(1, int(fps_input / 30))
            frame_times = []
            
            self.progress_update.emit(0, f"开始处理视频，共 {total_frames} 帧")
            
            while self.running and frame_count < total_frames:
                self.mutex.lock()
                if self.paused:
                    self.cond.wait(self.mutex)
                self.mutex.unlock()
                
                ret, frame = cap.read()
                if not ret:
                    break
                    
                should_process = (frame_count % skip_frames == 0)
                
                if should_process:
                    frame_start = time.time()
                    results = self.model(
                        frame,
                        conf=self.current_params['conf'],
                        iou=self.current_params['iou'],
                        max_det=self.current_params['max_det'],
                        device=self.device,
                        verbose=False
                    )
                    inference_time = (time.time() - frame_start) * 1000
                    
                    frame_times.append(inference_time)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    avg_time = sum(frame_times) / len(frame_times)
                    current_fps = 1000 / avg_time if avg_time > 0 else 0
                    
                    detections = self.parse_detections(results[0])
                    
                    # 绘制结果（返回可写的副本）
                    annotated_frame = self.draw_detections(
                        frame, results[0],
                        self.current_params['show_labels'],
                        self.current_params['show_conf']
                    )
                    
                    # 添加信息文字
                    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Inference: {inference_time:.1f}ms", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    processed_count += 1
                    
                    # 写入视频
                    writer.write(annotated_frame)
                    
                    # 发送帧
                    self.video_frame_ready.emit(annotated_frame, detections, current_fps, inference_time)
                else:
                    writer.write(frame)
                    self.video_frame_ready.emit(frame, [], self.fps, 0)
                
                frame_count += 1
                
                if frame_count % 30 == 0 or frame_count == total_frames:
                    progress = int((frame_count / total_frames) * 100)
                    self.progress_update.emit(progress, f"处理视频中: {frame_count}/{total_frames}")
                    
            cap.release()
            writer.release()
            
            # 保存检测信息
            info_path = os.path.join(save_dir, 'detection_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("YOLOv11 视频检测结果报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"源文件: {task['path']}\n")
                f.write(f"总帧数: {frame_count}\n")
                f.write(f"处理帧数: {processed_count}\n")
                f.write(f"输出视频: {output_path}\n")
                f.write("=" * 50 + "\n")
                
            self.progress_update.emit(100, f"视频处理完成，结果保存至: {save_dir}")
            
        except Exception as e:
            self.error_occurred.emit(f"视频检测失败: {str(e)}")
        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()
            self.finished.emit()
            
    def process_single_frame_task(self):
        task = self.current_task
        self.current_task = None
        
        try:
            if task.get('params'):
                self.update_params(**task['params'])
                
            frame = task['frame']
            
            start_time = time.time()
            results = self.model(
                frame,
                conf=self.current_params['conf'],
                iou=self.current_params['iou'],
                max_det=self.current_params['max_det'],
                device=self.device,
                verbose=False
            )
            inference_time = (time.time() - start_time) * 1000
            
            self.frame_times.append(inference_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            avg_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1000 / avg_time if avg_time > 0 else 0
            
            detections = self.parse_detections(results[0])
            
            # 绘制结果（返回可写的副本）
            annotated_frame = self.draw_detections(
                frame, results[0],
                self.current_params['show_labels'],
                self.current_params['show_conf']
            )
            
            # 添加信息文字
            cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Inference: {inference_time:.1f}ms", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.video_frame_ready.emit(annotated_frame, detections, self.fps, inference_time)
            
        except Exception as e:
            print(f"帧检测失败: {e}")
            
    def parse_detections(self, result):
        detections = []
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                detections.append({
                    'class': int(boxes.cls[i]),
                    'label': result.names[int(boxes.cls[i])],
                    'confidence': float(boxes.conf[i]),
                    'bbox': boxes.xyxy[i].cpu().numpy().tolist()
                })
        return detections
        
    def draw_detections(self, image, result, show_labels=True, show_conf=True):
        """绘制检测结果 - 返回可写的副本"""
        # 创建可写的副本
        annotated = image.copy()
        
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                label = result.names[cls_id]
                
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if show_labels or show_conf:
                    text = ""
                    if show_labels:
                        text += label
                    if show_conf:
                        if text:
                            text += f" {conf:.2f}"
                        else:
                            text += f"{conf:.2f}"
                            
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)
                    cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
        return annotated
        
    def save_result(self, image, result_type='image', source_path=None):
        """保存检测结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if result_type == 'image':
            save_dir = os.path.join('runs', 'image_results')
            os.makedirs(save_dir, exist_ok=True)
            
            # 使用原文件名
            if source_path:
                original_name = os.path.basename(source_path)
                name_without_ext = os.path.splitext(original_name)[0]
                save_path = os.path.join(save_dir, f'{name_without_ext}_{timestamp}.jpg')
            else:
                save_path = os.path.join(save_dir, f'result_{timestamp}.jpg')
            
            cv2.imwrite(save_path, image)
            return save_path
        else:
            save_dir = os.path.join('runs', 'results')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'result_{timestamp}.jpg')
            cv2.imwrite(save_path, image)
            return save_path
            
    def pause(self):
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
    def resume(self):
        self.mutex.lock()
        self.paused = False
        self.cond.wakeAll()
        self.mutex.unlock()
        
    def stop(self):
        self.running = False
        self.resume()