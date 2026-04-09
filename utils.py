import pandas as pd
import time
import os

class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0

    def update(self):
        """Calculates running FPS to assess hardware performance"""
        cur_time = time.time()
        self.fps = 1 / (cur_time - self.prev_time + 1e-6)
        self.prev_time = cur_time
        return self.fps

def generate_csv_report(counts, total, output_path="outputs/fruit_counts_report.csv"):
    """
    Generates a structured breakdown of detections. 
    Crucial for reporting mechanisms in mini-projects.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = []
    for fruit, count in counts.items():
        data.append({"Category": fruit, "Detected Quantity": count})
        
    df = pd.DataFrame(data)
    df = pd.concat([df, pd.DataFrame([{"Category": "TOTAL", "Detected Quantity": total}])], ignore_index=True)
    df.to_csv(output_path, index=False)
    return output_path
