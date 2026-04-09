import cv2
import numpy as np

class LineCrossingTracker:
    def __init__(self, line_position=0.5, orientation='horizontal'):
        self.line_position = line_position
        self.orientation = orientation
        self.track_history = {}  # Store previous coordinates to measure movement trajectory
        self.crossed_ids = set() # Avoid double counts
        self.counts = {}         # Dictionary Counter for dynamic tracking per category
        self.total_count = 0

    def update(self, frame, result):
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Design a highly visible detection intersection line
        if self.orientation == 'horizontal':
            line_y = int(height * self.line_position)
            cv2.line(annotated_frame, (0, line_y), (width, line_y), (255, 50, 50), 3)
            # Add line label
            cv2.putText(annotated_frame, "COUNTING LINE", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
        else:
            line_x = int(width * self.line_position)
            cv2.line(annotated_frame, (line_x, 0), (line_x, height), (255, 50, 50), 3)

        # Handle tracking algorithms output natively
        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            names = result.names

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = names.get(cls_id, str(cls_id))

                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append((cx, cy))
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)

                # Draw track trail for cool visualization
                points = np.hstack(self.track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 0), thickness=2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                # Line crossing logic (Increment counter)
                if len(self.track_history[track_id]) >= 2 and track_id not in self.crossed_ids:
                    prev_cx, prev_cy = self.track_history[track_id][-2]
                    curr_cx, curr_cy = self.track_history[track_id][-1]

                    crossed = False
                    if self.orientation == 'horizontal':
                        if (prev_cy < line_y <= curr_cy) or (prev_cy > line_y >= curr_cy):
                            crossed = True
                    else:
                        if (prev_cx < line_x <= curr_cx) or (prev_cx > line_x >= curr_cx):
                            crossed = True
                    
                    if crossed:
                        self.crossed_ids.add(track_id)
                        self.counts[class_name] = self.counts.get(class_name, 0) + 1
                        self.total_count += 1
                        
        return annotated_frame, self.counts, self.total_count
