# Real-Time Fruit Detection Engine (Machine Vision Project)

This project has been heavily upgraded to serve as a high-quality Machine Vision evaluation system. It replaces default generic models with a framework designed specifically to load custom-trained weights (`best.pt`), vastly improving metric accuracy.

## Upgrades Made
1. **Module Architecture Flattening:** Cleaned the application logic strictly into `main.py`, `detection.py`, `counting.py`, and `utils.py`.
2. **Sleek Interface:** The Streamlit dashboard was redesigned with a 2-column layout, embedded status metrics, dynamic alerting, and gradient bar charts to ensure presentation quality.
3. **Accuracy Focus:** Completely deprecated YOLOv8 COCO dependencies. Inference now explicitly defaults to extracting features against your custom dataset definitions via `best.pt`.

## Project Structure
```text
Fruit_Detection_System/
├── detection.py      # Core detection and intersection logic mapping
├── counting.py       # Object collision trackers (SORT algorithms)
├── utils.py          # FPS metric compilation and Data Export (CSV)
├── data/
│   └── dataset.yaml  # Dataset classes structure
├── main.py           # The robust 2-column Streamlit analytical dashboard
├── train.py          # Training interface geared to run 30-50 epochs
├── predict.py        # High-speed CLI execution script
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## How to Run the Visual Dashboard
```bash
py -3.11-64 -m streamlit run main.py
```

## How to execute Custom Training
```bash
py -3.11-64 train.py --data data/dataset.yaml --epochs 50 --model yolov8s.pt
```

## How to test headless CLI (Webcam)
```bash
py -3.11-64 predict.py --source 0 --model best.pt
```
