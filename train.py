import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/gd-yolov8.yaml', task='detect') # select your model.yaml path
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                patience=10,
                close_mosaic=10,
                workers=4,
                device='cuda',
                # optimizer='SGD', # using SGD 
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='FireDET',
                name='test1'
                )
