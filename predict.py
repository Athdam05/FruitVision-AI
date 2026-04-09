from ultralytics import YOLO
import argparse

def evaluate_source(source_path, model_path='best.pt', save=True):
    """
    Runs headless inference using the highly accurate custom trained model.
    """
    # Specifically load the finetuned weights instead of COCO
    print(f"Loading custom weights from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"CRITICAL: Failed to load {model_path}. Please train the model first or provide a valid path. Error: {e}")
        return
    
    # Analyze stream/file
    results = model(source_path, save=save, project="outputs", name="cli_predict_run")
    
    if save:
        print("Inference results and marked files saved to outputs/cli_predict_run")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Headless Fruit Detection")
    parser.add_argument("--source", type=str, required=True, help="Image, Video or 0 (Webcam)")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to your custom best.pt weights")
    parser.add_argument("--no-save", action="store_true", help="Disable output saving to disk")
    
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    evaluate_source(source, model_path=args.model, save=not args.no_save)
