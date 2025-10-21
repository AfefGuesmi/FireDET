import warnings

warnings.filterwarnings('ignore')
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

from ultralytics import YOLO

if __name__ == '__main__':
    # Define checkpoint paths
    weights_dir = 'pretrained'
    best_pt = os.path.join(weights_dir, 'best.pt')
    last_pt = os.path.join(weights_dir, 'last.pt')

    # Check if best.pt exists, otherwise use last.pt
    if os.path.isfile(best_pt):
        print(f"✓ Resuming from best checkpoint: {best_pt}")
        checkpoint_path = best_pt
    elif os.path.isfile(last_pt):
        print(f"✓ Resuming from last checkpoint: {last_pt}")
        checkpoint_path = last_pt
    else:
        print("✗ No checkpoint found, starting from pre-trained weights")
        checkpoint_path = 'yolov8n.pt'

    # Load model from appropriate checkpoint
    model = YOLO(checkpoint_path)

    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                patience=10,
                close_mosaic=10,
                workers=0,
                device='gpu',
                project='FireDET',
                name='test1',
                resume=True
                )