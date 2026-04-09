import cv2
from ultralytics import YOLO

class FruitDetector:
    def __init__(self, model_path='best.pt'):
        """
        Loads the custom-trained model (best.pt) to ensure the system 
        is exclusively tuned for the fruit dataset.
        """
        try:
            self.model = YOLO(model_path)
            self.labels = self.model.names
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            self.model = None
            self.labels = {}

    def detect(self, frame, conf_threshold=0.25, iou_threshold=0.20, imgsz=1280):
        """Standard detection. Low iou = allows tightly packed fruits. High imgsz improves accuracy."""
        if not self.model: return []
        results = self.model(frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)
        return results[0]

    def detect_sahi(self, frame, slice_size=640, overlap=0.25, conf_threshold=0.25, iou_threshold=0.20):
        """Advanced detection using overlapping patches to perfectly find multiple densely packed fruits."""
        if not self.model: return [], [], []
        
        h, w = frame.shape[:2]
        step = int(slice_size * (1 - overlap))
        
        all_boxes, all_confs, all_cls = [], [], []
        
        y = 0
        while y < h:
            x = 0
            while x < w:
                y2 = min(y + slice_size, h)
                x2 = min(x + slice_size, w)
                
                y1 = y
                x1 = x
                if y2 - y1 < slice_size: y1 = max(0, y2 - slice_size)
                if x2 - x1 < slice_size: x1 = max(0, x2 - slice_size)
                
                tile = frame[y1:y2, x1:x2]
                
                res = self.model(tile, conf=conf_threshold, imgsz=slice_size, verbose=False)[0]
                for box in res.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                    all_confs.append(float(box.conf[0]))
                    all_cls.append(int(box.cls[0]))
                    
                x += step
                if x1 == max(0, w - slice_size) and x != 0: break
            y += step
            if y1 == max(0, h - slice_size) and y != 0: break
            
        if not all_boxes: return [], [], []
        
        import torch
        from torchvision.ops import nms
        
        keep_indices = nms(
            torch.tensor(all_boxes, dtype=torch.float32), 
            torch.tensor(all_confs, dtype=torch.float32), 
            iou_threshold
        )
        
        final_boxes, final_confs, final_cls = [], [], []
        for idx in keep_indices.tolist():
            final_boxes.append(all_boxes[idx])
            final_confs.append(all_confs[idx])
            final_cls.append(all_cls[idx])
            
        return final_boxes, final_confs, final_cls

    def draw_boxes(self, frame, result):
        """Draw bounding boxes natively via OpenCV for maximum control"""
        annotated = frame.copy()
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.labels.get(cls_id, str(cls_id))
            self._draw_box(annotated, x1, y1, x2, y2, label, conf)
        return annotated

    def draw_manual_boxes(self, frame, boxes, confs, clss):
        annotated = frame.copy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            label = self.labels.get(clss[i], str(clss[i]))
            self._draw_box(annotated, x1, y1, x2, y2, label, confs[i])
        return annotated

    def _draw_box(self, img, x1, y1, x2, y2, label, conf):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 128), 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + tw, y1), (0, 255, 128), -1)
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
