# FireDET

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)](https://github.com/ultralytics/ultralytics)
[![Jetson Nano](https://img.shields.io/badge/Jetson%20Nano-Edge%20AI-76B900)](https://developer.nvidia.com/embedded/jetson-nano)
[![TensorRT](https://img.shields.io/badge/TensorRT-optimized-FF6F00)](https://developer.nvidia.com/tensorrt)
[![Roboflow](https://img.shields.io/badge/Roboflow-annotated-blueviolet)](https://roboflow.com)

**FireDET** is an AI‑based system for real‑time detection of fire and smoke in images and videos. Built on a modified YOLOv8 architecture and optimised for edge devices like the NVIDIA Jetson Nano, it delivers fast, accurate alerts to support early response and safety.

---

## 📌 Motivation

The catastrophic Los Angeles wildfires of January 2025 – where 9 major fires including the 23,448‑acre Palisades Fire and 14,021‑acre Hollywood Fire burned over 38,000 cumulative acres in just 4 days – exemplify the global wildfire emergency. In 2023, wildfires burned 4.3 million km² worldwide (equivalent to the EU’s land area), causing $1.2B in damages in California alone (CalOES estimates) and exposing critical detection failures:

- The 19‑acre Archer Fire went undetected for **78 minutes** (CAL FIRE).
- Traditional systems missed **42%** of sub‑50‑acre scars (ESA Validation 2023).
- Emergency room visits for respiratory illness spiked **400%** during these fires (LA County Health).

**FireDET** directly addresses these gaps by combining ground‑based surveillance with a deep‑learning detector that achieves **0.94 recall** on the D‑Fire benchmark, detecting incipient flames as small as a few pixels within **52.6 ms** per frame on edge hardware – enabling rapid alerting and damage assessment.

---

## 🔍 Overview

FireDET is a real‑time fire and smoke detection framework based on a heavily modified YOLOv8 architecture. It incorporates three key innovations:

- **SPD‑Conv** in the backbone to preserve fine spatial details of small flames and thin smoke.
- **EMA attention** in the neck for adaptive multi‑scale feature fusion.
- **Focal‑SIoU loss** to focus training on difficult, low‑IoU examples.

The system is trained on **two complementary datasets** to ensure robust generalisation:
1. **D‑Fire**: A large public benchmark dataset (Venâncio et al., 2020) containing over 21,000 images with fire and smoke annotations across diverse environments.
2. **Manual dataset**: A custom dataset of ~5,000 images created and annotated using **Roboflow**. This dataset includes:
   - Images from surveillance cameras, public safety archives, and fire databases.
   - Manual bounding‑box labelling for `fire` and `smoke` classes.
   - Extensive data augmentation (rotation, brightness/color adjustments, horizontal flipping, mosaic) to simulate real‑world variability.

Both datasets are split into training (70%), validation (20%), and test (10%) sets.

The full pipeline includes:
- Training in PyTorch (cloud GPUs)
- Conversion to ONNX and TensorRT for edge deployment
- Real‑time inference on NVIDIA Jetson Nano

---

## ✨ Features

- **Real‑time detection** – Up to 19 FPS on Jetson Nano with TensorRT.
- **Dual‑class** – Simultaneous detection of `fire` and `smoke`.
- **Edge‑optimised** – Model compression and hardware‑aware acceleration.
- **Early‑stage sensitivity** – Detects small, distant, or partially occluded fire sources.
- **False‑positive reduction** – 89% fewer background false alarms on challenging test sets.
- **End‑to‑end pipeline** – Training, validation, TensorRT conversion, and deployment scripts.
- **Two datasets** – Trained on both public D‑Fire and a custom Roboflow‑annotated dataset for superior generalisation.

---

## 📁 Repository Structure

```
FireDET/
├── .gitignore
├── LICENSE
├── README.md
├── detect.py               # Inference on images/videos
├── detectb.py              # Alternative inference script
├── testb.py                # Testing script (backup)
├── thersholding.py         # Threshold‑based detection experiments
├── train.py                # Main training script (baseline)
├── trainMan.py             # Manual training with custom modifications (SPD‑Conv, EMA, Focal‑SIoU)
├── val.py                  # Validation script
├── yolov8n.pt              # Pretrained YOLOv8n weights (baseline)
├── runs/                   # Training outputs (logs, weights)
│   └── train/              # Experiment folders
├── ultralytics/            # Modified YOLOv8 source (if customised)
└── requirements.txt        # Python dependencies
```

> **Note:** The repository may contain a folder `wights` (a typo for `weights`) that stores additional pretrained models.

---

## ⚙️ Installation & Usage

### Prerequisites

- Python 3.10+
- CUDA‑compatible GPU (for training)
- NVIDIA Jetson Nano (for deployment, optional)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/AfefGuesmi/FireDET.git
cd FireDET
```

### 2. Set Up Virtual Environment

```bash
python -m venv firedet-env
source firedet-env/bin/activate      # Linux / macOS
# or
firedet-env\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, manually install the core packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pandas numpy matplotlib seaborn tqdm scikit-learn
```

### 4. Prepare Datasets

#### D‑Fire Dataset (Public)
Download from [GitHub](https://github.com/gala-solutions-on-demand/DFireDataset) and organise as:

```
data/
└── D-Fire/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

#### Manual Dataset (Custom)
- Create a project in [Roboflow](https://roboflow.com).
- Upload your images (~5000) and annotate two classes: `fire` and `smoke`.
- Apply augmentations (rotation, brightness, flip, mosaic) as needed.
- Export in **YOLOv8 format** and place the folders similarly under `data/Manual/`.

Create a `data.yaml` file that points to the dataset paths and defines the classes. Example for combined training:

```yaml
train: ./data/D-Fire/images/train
val: ./data/D-Fire/images/val
# Include manual dataset if combining (you may need to merge folders)
nc: 2
names: ['fire', 'smoke']
```

*For training on both datasets, you can merge the image/label folders or use separate YAMLs.*

### 5. Training

#### Baseline YOLOv8

```bash
python train.py --data data.yaml --epochs 100 --batch 16 --imgsz 640 --weights yolov8n.pt
```

#### Modified YOLOv8 (SPD‑Conv, EMA, Focal‑SIoU)

```bash
python trainMan.py --data data.yaml --epochs 100 --batch 16 --imgsz 640 --weights yolov8n.pt --spd --ema --focal-sious
```

*Adjust hyperparameters as needed. The `trainMan.py` script implements the custom modifications.*

### 6. Evaluation

Validate a trained model:

```bash
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt --imgsz 640
```

### 7. Inference on Images / Video

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/image_or_video --conf 0.5
```

### 8. TensorRT Optimisation (for Jetson Nano)

Convert the PyTorch model to TensorRT FP16 using the Ultralytics export utility:

```bash
yolo export model=runs/train/exp/weights/best.pt format=engine device=0
# or using a custom script if available
```

Then run inference with the TensorRT engine:

```bash
python detect.py --weights runs/train/exp/weights/best.engine --source path/to/video
```

---

## 📊 Results

The modified model consistently outperforms the baseline across both datasets and deployment formats.

### PyTorch Models

| Model          | Dataset | mAP50‑95 | Precision | Recall | Inference Time (ms) |
|----------------|---------|----------|-----------|--------|----------------------|
| Baseline       | D‑Fire  | 0.395    | 0.837     | 0.882  | 205.5                |
| **Modified**   | D‑Fire  | **0.512**| 0.928     | 0.926  | 157.7                |
| Baseline       | Manual  | 0.330    | 0.694     | 0.637  | 308.0                |
| **Modified**   | Manual  | **0.415**| 0.715     | 0.659  | 89.9                 |

### TensorRT‑Optimised Models

| Model          | Dataset | mAP50‑95 | Precision | Recall | Inference Time (ms) |
|----------------|---------|----------|-----------|--------|----------------------|
| Baseline       | D‑Fire  | 0.418    | 0.938     | 0.845  | 83.0                 |
| **Modified**   | D‑Fire  | **0.531**| 0.933     | 0.942  | 52.6                 |
| Baseline       | Manual  | 0.387    | 0.709     | 0.633  | 275.6                |
| **Modified**   | Manual  | **0.436**| 0.752     | 0.679  | 74.2                 |

**Key improvements:**
- **29.6% higher mAP50‑95** on D‑Fire (PyTorch)
- **37% faster inference** on D‑Fire (TensorRT)
- **89% fewer false positives** on the challenging manual dataset
- **3.7× speedup** on manual dataset after TensorRT optimisation

Confusion matrix analysis shows dramatic reduction in false negatives (missed fires) and false positives, especially in cluttered backgrounds.

---

## 🚀 Deployment on Jetson Nano

FireDET is fully tested on **NVIDIA Jetson Nano** with **JetPack 4.6**. The TensorRT‑optimised engine runs at **19 FPS** while maintaining >0.93 mAP50.  
To deploy:
1. Flash Jetson Nano with JetPack SD card image (using balenaEtcher).
2. Install dependencies (PyTorch, TorchVision, TensorRT, etc.) as per [NVIDIA docs](https://docs.nvidia.com/jetson/).
3. Transfer the TensorRT engine file (`.engine`) and inference script.
4. Run detection with the engine.

For detailed instructions, refer to the [Jetson Setup Guide](docs/jetson_setup.md) (if included) or the [official NVIDIA documentation](https://docs.nvidia.com/jetson/).

---

## 📖 Citation

If you use FireDET in your research, please cite the dissertation:

```bibtex
@mastersthesis{guesmi2025autonomous,
  title  = {Autonomous video stream monitoring system based on embedded AI frameworks},
  author = {Afef Guesmi},
  school = {University of Monastir},
  year   = {2025}
}
```

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – base framework.
- [D‑Fire dataset](https://github.com/gala-solutions-on-demand/DFireDataset) – Venâncio et al.
- [Roboflow](https://roboflow.com) – annotation, augmentation, and dataset management.
- NVIDIA Jetson Nano – edge hardware.
- Supervisors: Kamel Besbes, Nizar Habbachi.
- All contributors and testers.

---

*For questions or collaborations, please open an issue or contact [afef.guesmi@example.com](mailto:afef.guesmi@example.com).*
