import warnings

warnings.filterwarnings('ignore')
import os
import tempfile

# Set environment variables to control where temporary files are stored
os.environ["WANDB_MODE"] = "disabled"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

# Create a custom temporary directory with a short path
temp_dir = tempfile.mkdtemp(prefix='FD_', dir='C:\\')
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

from ultralytics import YOLO

if __name__ == '__main__':
    # Create a simple directory structure to avoid path length issues
    base_dir = 'C:\\FD'
    project_dir = os.path.join(base_dir, 'FireDET')
    weights_dir = os.path.join(project_dir, 'test1', 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Check if we have a checkpoint to resume from
    checkpoint_path = os.path.join(weights_dir, 'best.pt')

    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        resume = True
    else:
        print("Starting training from pre-trained weights")
        model = YOLO('yolov8n.pt')
        resume = False

    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=320,  # Reduced from 640 to save memory
                epochs=100,
                batch=2,
                patience=10,
                close_mosaic=10,
                workers=0,
                device='cpu',
                project=base_dir,  # Use the shorter base path
                name='FireDET',  # Simple name
                resume=resume  # Resume from checkpoint if available
                )