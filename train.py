import argparse
from ultralytics import YOLO

def train_model(data_yaml, epochs=50, imgsz=640, model_name='yolov8s.pt'):
    """
    Trains a custom YOLOv8 model using a specific dataset configuration.
    Outputs the final highly accurate weights to 'best.pt'
    """
    # Start with a pretrained base to utilize transfer learning 
    model = YOLO(model_name)
    
    # Train heavily on our specific custom dataset
    results = model.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=imgsz, 
        project="outputs", 
        name="custom_fruit_train"
    )
    print("Training Complete. The newly validated weights 'best.pt' are stored in the outputs directory.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train a custom YOLOv8 Fruit Detector.")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Location of dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for accuracy maximization (30-50 recommended)")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base pretrained model.")
    
    args = parser.parse_args()
    print(f"Initializing finetuning: Dataset {args.data} | Epochs {args.epochs}")
    train_model(args.data, args.epochs, model_name=args.model)
