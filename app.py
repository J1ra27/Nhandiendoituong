import cv2
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.backends.cudnn as cudnn
import numpy as np

class VideoDetectionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phát hiện đối tượng với YOLOv5")
        self.root.geometry("500x150")
        
        self.video_path = tk.StringVar()
        self.create_widgets()
        self.setup_gpu()
        self.object_counts = {}
        
        # Định nghĩa màu cố định cho mỗi class
        self.class_colors = {
            'person': (0, 255, 0),     # Xanh lá
            'bicycle': (255, 0, 0),    # Xanh dương
            'car': (0, 0, 255),        # Đỏ
            'motorcycle': (255, 255, 0),# Vàng
            'bus': (255, 0, 255),      # Tím
            'truck': (0, 255, 255),    # Cam
            'traffic light': (128, 0, 0),  # Nâu đỏ
            'stop sign': (0, 128, 0),   # Xanh đậm
            'bench': (0, 0, 128),       # Đỏ đậm
            'bird': (128, 128, 0),      # Olive
            'cat': (128, 0, 128),       # Tím đậm
            'dog': (0, 128, 128),       # Xanh ngọc
            # Thêm màu cho các class khác...
        }
        
        # Màu mặc định cho các class chưa được định nghĩa
        self.default_color = (192, 192, 192)  # Xám

    def setup_gpu(self):
        if torch.cuda.is_available():
            cudnn.benchmark = True
            cudnn.deterministic = False
            torch.cuda.empty_cache()

    def create_widgets(self):
        # Frame cho phần input
        input_frame = tk.LabelFrame(self.root, text="Chọn Video", padx=20, pady=15)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Label cho đường dẫn video
        tk.Label(input_frame, 
                text="Đường dẫn video:", 
                font=('Arial', 10)).pack(side=tk.LEFT)
        
        # Entry cho đường dẫn video
        entry = tk.Entry(input_frame, 
                        textvariable=self.video_path, 
                        width=50,
                        font=('Arial', 10))
        entry.pack(side=tk.LEFT, padx=10)
        
        # Nút Duyệt
        browse_button = tk.Button(
            input_frame, 
            text="Duyệt",
            command=self.browse_file,
            width=10,
            height=1,
            font=('Arial', 10, 'bold'),
            bg='#4a90e2',  # Màu xanh dương
            fg='white',    # Chữ màu trắng
            relief=tk.RAISED,
            cursor="hand2"  # Con trỏ chuột kiểu bàn tay
        )
        browse_button.pack(side=tk.LEFT)

        # Frame cho các nút điều khiển
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=15)

        # Nút bắt đầu xử lý
        process_button = tk.Button(
            button_frame, 
            text="Bắt đầu xử lý",
            command=self.process_video,
            width=15,
            height=1,
            font=('Arial', 10, 'bold'),
            bg='#2ecc71',  # Màu xanh lá
            fg='white',    # Chữ màu trắng
            relief=tk.RAISED,
            cursor="hand2"
        )
        process_button.pack(side=tk.LEFT, padx=5)

        # Thêm hiệu ứng hover cho các nút
        def on_enter(e):
            e.widget['bg'] = '#3498db' if e.widget['text'] == "Duyệt" else '#27ae60'

        def on_leave(e):
            e.widget['bg'] = '#4a90e2' if e.widget['text'] == "Duyệt" else '#2ecc71'

        # Bind các sự kiện hover
        browse_button.bind("<Enter>", on_enter)
        browse_button.bind("<Leave>", on_leave)
        process_button.bind("<Enter>", on_enter)
        process_button.bind("<Leave>", on_leave)

        # Điều chỉnh kích thước cửa sổ
        self.root.geometry("800x200")  # Tăng kích thước cửa sổ

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)

    def get_color(self, class_name):
        """Lấy màu cho class, trả về màu mặc định nếu chưa được định nghĩa"""
        return self.class_colors.get(class_name, self.default_color)

    def process_video(self):
        if not self.video_path.get():
            messagebox.showerror("Lỗi", "Vui lòng chọn video!")
            return

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cpu':
                messagebox.showwarning("Cảnh báo", "Không tìm thấy GPU. Đang sử dụng CPU!")
            else:
                messagebox.showinfo("Thông báo", f"Đang sử dụng GPU: {torch.cuda.get_device_name()}")
            
            # Tải model YOLOv5x
            model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
            model.to(device)
            model.eval()
            
            # Cấu hình model
            model.conf = 0.5  # Ngưỡng confidence
            model.iou = 0.45  # Ngưỡng NMS IOU
            model.classes = None  # Phát hiện tất cả classes
            
            cap = cv2.VideoCapture(self.video_path.get())
            if not cap.isOpened():
                raise Exception("Không thể mở tệp video.")

            fps_counter = 0
            prev_time = time.time()
            fps_to_display = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Cập nhật FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - prev_time > 1.0:
                    fps_to_display = fps_counter
                    fps_counter = 0
                    prev_time = current_time

                try:
                    # Reset object counts cho mỗi frame
                    self.object_counts.clear()
                    
                    # Dự đoán với YOLOv5
                    results = model(frame)
                    
                    # Xử lý kết quả..
                    display_frame = frame.copy()
                    
                    if len(results.xyxy[0]) > 0:
                        for det in results.xyxy[0]:
                            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                            if conf > 0.5:  # Ngưỡng tin cậy
                                # Chuyển đổi tọa độ sang int
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                
                                # Lấy tên class trực tiếp từ model
                                class_name = model.names[int(cls)]
                                
                                # Cập nhật số lượng đối tượng
                                self.object_counts[class_name] = self.object_counts.get(class_name, 0) + 1
                                
                                # Lấy màu cố định cho class
                                color = self.get_color(class_name)
                                
                                # Vẽ bounding box với độ dày 2
                                cv2.rectangle(display_frame, 
                                            (x1, y1), 
                                            (x2, y2), 
                                            color, 2)
                                
                                # Vẽ label với confidence
                                label = f"{class_name} {conf:.2f}"
                                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                
                                # Vẽ nền cho label với cùng màu của bounding box
                                cv2.rectangle(display_frame,
                                            (x1, y1 - t_size[1] - 10),
                                            (x1 + t_size[0], y1),
                                            color, -1)
                                            
                                # Vẽ text màu trắng
                                cv2.putText(display_frame, 
                                          label,
                                          (x1, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.7, (255, 255, 255), 2)

                    # Vẽ thống kê với màu tương ứng của mỗi class
                    y_offset = 30
                    for class_name, count in self.object_counts.items():
                        color = self.get_color(class_name)
                        stats_text = f"{class_name}: {count}"
                        cv2.putText(display_frame,
                                  stats_text,
                                  (10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6,
                                  color,
                                  2)
                        y_offset += 30

                    # Vẽ FPS với màu xanh lá
                    cv2.putText(display_frame,
                              f"FPS: {int(fps_to_display)}",
                              (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.6,
                              (0, 255, 0),
                              2)

                    # Hiển thị frame
                    cv2.imshow("YOLOv5x Object Detection", display_frame)

                except Exception as e:
                    print(f"Lỗi khi xử lý frame: {e}")
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoDetectionApp()
    app.run()