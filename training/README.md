# Chess Piece Detection -- Training Pipeline

Automated end-to-end pipeline that collects chess-piece detection datasets,
validates them, prepares YOLO-format splits, augments training data, and
fine-tunes a YOLOv8s model. The trained `chess.pt` is copied to the project
root for direct use by the main application.

## Quick Start

```bash
cd training

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (collect -> validate -> prepare -> augment -> train -> eval -> integrate)
python run_pipeline.py

# Or run individual phases
python collect_data.py          # Phase 1:  download datasets
python validate_dataset.py      # Phase 1b: validate raw data
python prepare_dataset.py       # Phase 2:  dedup, normalize, split
python augment_data.py           # Phase 3:  augment training split
python train.py                  # Phase 4:  train YOLOv8s
python validate.py               # Phase 5:  evaluate model
python integrate.py              # Phase 6:  integration check
```

## Pipeline Runner Options

```bash
python run_pipeline.py                        # run everything
python run_pipeline.py --skip collect          # skip download (use existing data)
python run_pipeline.py --only train eval       # run specific phases only
python run_pipeline.py --from prepare          # start from a specific phase
python run_pipeline.py --force                 # overwrite existing dataset
python run_pipeline.py --force-collect         # re-download all sources
```

## Dataset Collection (Phase 1)

The collector downloads chess-piece datasets from multiple public sources
**without any Roboflow dependency**. Each source is attempted independently;
failures are logged and automatically skipped.

### Download strategies (tried in order per source)

1. **`requests`** with retry logic + exponential backoff
2. **`curl`** subprocess fallback
3. **`git clone`** fallback (for repository sources)

### SSL handling for Windows

- Uses `certifi` CA bundle when available
- Falls back to system certificates
- Last resort: unverified SSL (Windows only)

### Collector options

```bash
python collect_data.py                          # try all sources
python collect_data.py --sources github_chess_erdogant hf_chess_pieces_francesco
python collect_data.py --list                   # list available sources
python collect_data.py --force                  # re-download existing data
python collect_data.py --validate-only          # validate without downloading
```

### Adding custom sources

Place YOLO-format data (images + `.txt` labels) in:

```
training/raw_data/<your_dataset_name>/
    images/
        img001.jpg
        img002.jpg
    labels/
        img001.txt
        img002.txt
```

Then run `python prepare_dataset.py --force` to incorporate it.

## Dataset Validation (Phase 1b)

Validates downloaded data before preparation:

- **Image-label pair matching** -- every image should have a `.txt` label
- **YOLO format correctness** -- `class_id cx cy w h`, values normalised 0-1
- **Class coverage** -- checks all 12 chess classes are represented
- **Corrupt image detection** -- OpenCV read check
- **Detailed reporting** with counts and issue lists

```bash
python validate_dataset.py                     # validate raw_data/
python validate_dataset.py --prepared          # validate prepared dataset/
python validate_dataset.py --strict            # require all 12 classes
python validate_dataset.py --path /some/dir    # validate custom directory
```

## Folder Structure

```
training/
├── config.py              # paths, classes, hyperparams
├── dataset.yaml           # YOLOv8 data config
├── run_pipeline.py        # pipeline orchestrator
├── collect_data.py        # Phase 1:  automated dataset downloader
├── validate_dataset.py    # Phase 1b: dataset validation
├── prepare_dataset.py     # Phase 2:  dedup, normalize, split
├── augment_data.py        # Phase 3:  Albumentations augmentation
├── train.py               # Phase 4:  YOLOv8s fine-tuning
├── validate.py            # Phase 5:  mAP, confusion matrix
├── inference_test.py      # smoke-test inference
├── integrate.py           # Phase 6:  app compatibility check
├── requirements.txt       # training-specific dependencies
│
├── raw_data/              # (runtime) downloaded datasets
│   ├── <source_name>/
│   └── ...
├── dataset/               # (runtime) prepared YOLO splits
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── runs/                  # (runtime) training outputs
```

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

Class names must match `config/settings.py :: PIECE_TO_FEN` keys.

## Requirements

Core (required):
- `ultralytics>=8.4.0`
- `opencv-python>=4.13.0`
- `numpy>=2.4.0`
- `albumentations>=2.0.0`
- `requests>=2.31.0`
- `PyYAML>=6.0`
- `certifi>=2024.0.0`

Optional (for broader source support):
- `huggingface_hub>=0.20.0` -- HuggingFace snapshot downloads
- `datasets>=3.0.0` -- HuggingFace datasets library
- `kaggle>=1.6.0` -- Kaggle dataset downloads

## Tips

- **Resume training:** `python train.py --resume`
- **Custom epochs:** `python train.py --epochs 100 --batch 8`
- **GPU required:** training is extremely slow on CPU; use CUDA if available
- **Manual data:** drop YOLO-format folders in `raw_data/` and run from `--from prepare`
- **Check weights:** `python inference_test.py` after training to sanity-check detections
