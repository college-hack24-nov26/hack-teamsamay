# SUTRA Model Directory & Custom Weight Specification

This directory holds deep learning model weights for the **SUTRA Vision Engine**.

---

## 1. Supported Classes (Canonical Laboratory Taxonomy)

The SUTRA experiment verification engine tracks the following 6 core lab classes:
1. `hand` (Human astronaut hand/glove)
2. `vial_sealed` (Sample container with seal/cap intact)
3. `vial_open` (Sample container with cap removed / fluid accessible)
4. `pipette` (Liquid transfer tool / micropipette)
5. `tip_box` (Sterile pipette tip rack)
6. `waste_bin` (Liquid/solid disposal container)

---

## 2. Model Loading & Scientific Integrity Policy

> [!IMPORTANT]
> **Strict Anti-Fabrication Rule:** A generic pretrained COCO model (trained on chairs, dogs, cars, and forks) does **not** recognize space biology laboratory items.
> SUTRA will **never** claim an untrained or off-the-shelf COCO model is recognizing micropipettes or sample vials.

When custom ONNX weights are placed at `models/yolo_sutra.onnx`, the engine automatically loads them via **ONNX Runtime CPU**. If custom weights are not present, the system transparently activates `HeuristicLabDetector` (color & geometry contour analysis) and prints an explicit notification.

---

## 3. Training & Exporting Custom YOLO ONNX Weights

To train and deploy fine-tuned YOLOv8n or YOLO11n weights:

### Step 1: Annotate Dataset
Label images with the 6 canonical classes in YOLO format (`class_id x_center y_center width height`).

### Step 2: Train Model
```bash
yolo task=detect mode=train model=yolo11n.pt data=sutra_dataset.yaml epochs=100 imgsz=640 device=cpu
```

### Step 3: Export to ONNX (CPU-Optimized)
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=17 simplify=True
```

### Step 4: Deploy
Copy the exported ONNX model to this directory:
```bash
cp runs/detect/train/weights/best.onnx models/yolo_sutra.onnx
```

The engine will automatically detect and load `models/yolo_sutra.onnx` on next launch!

