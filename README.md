# Site Safety Inspector

Object detection web app for construction-site safety. A YOLOv8 model, fine-tuned on a PPE dataset, locates workers and checks for hard hats, masks and safety vests. Built with Ultralytics and Streamlit.

## Features

- Upload an image (JPG, PNG, WEBP) or choose a bundled sample
- Confidence threshold slider, NMS overlap slider and class filter
- Annotated image with bounding boxes and labels
- Table of detected objects with confidence scores, plus per-class summary
- Safety violation banner for classes prefixed with `NO-` (for example NO-Hardhat)
- Download the annotated image and results as CSV

## Project structure

```
ppe-detector/
  app.py                       Streamlit application
  requirements.txt             Python dependencies (CPU PyTorch)
  packages.txt                 System libraries for Streamlit Community Cloud
  .streamlit/config.toml       Theme and server settings
  models/best.pt               Trained weights (added after training)
  samples/                     Optional sample images for the demo
  notebooks/train_yolov8_ppe.ipynb   Training notebook
```

## Dataset

Construction Site Safety image dataset (Roboflow export, YOLO format), available on Kaggle:
`snehilsanyal/construction-site-safety-image-dataset-roboflow`

Classes: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle.

## Model

- Base: `yolov8n.pt` (COCO pretrained)
- Fine-tuning: 30 epochs, image size 640, batch 16
- Results: see the training notebook for the full epoch log and final mAP50

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# place the trained weights at models/best.pt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public), including `models/best.pt`.
2. Sign in at https://share.streamlit.io with GitHub.
3. Select Create app, choose the repository, branch `main`, main file `app.py`.
4. Under Advanced settings, select Python 3.11.
5. Deploy. The first build takes several minutes.
