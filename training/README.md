# Chess-Piece Detection — Training Pipeline

Automated dataset collection, preparation, augmentation, training, evaluation,
and integration pipeline for the YOLOv8 chess-piece detector.

---

## Quick Start

```bash
cd training

# 1. Install extra dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (collect → prepare → augment → train → eval → integrate)
python run_pipeline.py

# Or run individual phases:
python collect_data.py          # Phase 1: Download datasets
python prepare_dataset.py       # Phase 2: Dedup, normalise, format, split
python augment_data.py          # Phase 3: Data augmentation
python train.py                 # Phase 4: Train YOLOv8s
python validate.py              # Phase 5: Evaluate model
python inference_test.py --test-split --show   # Quick visual check
python integrate.py             # Phase 6: Verify app compatibility
```

---

## Folder Structure

```
training/
├── config.py              # Pipeline configuration (classes, paths, hyper-params)
├── dataset.yaml           # YOLOv8 data config
├── requirements.txt       # Extra Python dependencies
├── run_pipeline.py        # One-click full pipeline runner
│
├── collect_data.py        # Phase 1 — Dataset collection
├── prepare_dataset.py     # Phase 2 — Dedup + normalise + YOLO format + split
├── augment_data.py        # Phase 3 — Augmentations (rotation, blur, noise, …)
├── train.py               # Phase 4 — YOLOv8s fine-tuning
├── validate.py            # Phase 5 — mAP, precision, recall, confusion matrix
├── inference_test.py      # Phase 4 — Quick inference smoke-test
├── integrate.py           # Phase 6 — Verify compatibility with the main app
│
├── raw_data/              # (created) Downloaded source datasets
├── dataset/               # (created) Prepared YOLO dataset
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── runs/                  # (created) Training & evaluation outputs
```

---

## Phase Details

### Phase 1 — Dataset Collection (`collect_data.py`)

Downloads chess-piece images + annotations from:

| Source       | Auth Required                | Env Vars                          |
|-------------|------------------------------|-----------------------------------|
| Roboflow    | API key (free tier)          | `ROBOFLOW_API_KEY`                |
| Kaggle      | API key                      | `KAGGLE_USERNAME`, `KAGGLE_KEY`   |
| HuggingFace | None (public)                | —                                 |
| Direct URLs | None                         | —                                 |

You can also place datasets manually under `raw_data/<name>/` with standard
YOLO layout (`images/` + `labels/` folders, or images alongside `.txt` labels).

### Phase 2 — Dataset Preparation (`prepare_dataset.py`)

- Scans every sub-folder of `raw_data/`
- Detects class-name files (`classes.txt`, `obj.names`, `data.yaml`)
- Remaps class IDs to the canonical 12 chess-piece classes
- Removes duplicate images (perceptual hashing)
- Resizes all images to 640×640
- Splits into 70% train / 20% val / 10% test

### Phase 3 — Augmentation (`augment_data.py`)

Applied to the **train split only** (val/test stay clean):

| Augmentation          | Probability |
|-----------------------|-------------|
| Rotation (±15°)       | 50%         |
| Perspective distortion| 30%         |
| Affine shear          | 30%         |
| Brightness/contrast   | 50%         |
| Random shadows        | 30%         |
| Gaussian blur         | 30%         |
| Gaussian noise        | 30%         |
| Partial occlusion     | 30%         |

Default: 3 augmented copies per original image.

### Phase 4 — Training (`train.py`)

- Base model: `yolov8s.pt` (pretrained on COCO)
- Image size: 640
- Epochs: 50 (with early stopping, patience=10)
- Outputs best weights as `chess.pt` in the project root

### Phase 5 — Evaluation (`validate.py`)

Generates:
- **mAP50** and **mAP50-95**
- **Precision** and **Recall**
- **Per-class AP**
- **Confusion matrix** (saved as PNG)
- **Sample prediction visualisations**

### Phase 6 — Integration (`integrate.py`)

Verifies:
- `chess.pt` exists and loads correctly
- All model class names are in `PIECE_TO_FEN`
- Class ordering is compatible
- Sanity inference produces detections

---

## 12 Chess-Piece Classes

| ID | Class Name    | FEN |
|----|---------------|-----|
| 0  | white-pawn    | P   |
| 1  | white-knight  | N   |
| 2  | white-bishop  | B   |
| 3  | white-rook    | R   |
| 4  | white-queen   | Q   |
| 5  | white-king    | K   |
| 6  | black-pawn    | p   |
| 7  | black-knight  | n   |
| 8  | black-bishop  | b   |
| 9  | black-rook    | r   |
| 10 | black-queen   | q   |
| 11 | black-king    | k   |

---

## Tips

- **Manual data**: Drop any YOLO-format dataset into `raw_data/` with a
  `classes.txt` listing its class names. The pipeline auto-maps them.
- **Resume training**: `python train.py --resume`
- **Evaluate on validation set**: `python validate.py --split val`
- **Skip slow phases**: `python run_pipeline.py --skip collect`
- **Force re-prepare**: `python run_pipeline.py --force`
