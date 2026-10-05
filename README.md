# YOLO Dataset Trainer (FastAPI)

## Run
```
pip install -r requirements.txt
python main.py
```
App runs at http://localhost:5000

## Flow
1. **Class Separator** (top of page) - upload one annotated dataset zip.
   Images are grouped into per-class folders (class index rewritten to 0
   in each label file) with a Download button per class.
2. **Positive (object detection) dataset** - upload multiple zip datasets
   at once, each checked individually (structure + image/label match).
   Enter a class name - it's compared against every uploaded folder's
   data.yaml; mismatches can be fixed in one click across all folders.
   All uploaded folders are then combined into a single dataset folder,
   named and downloaded.
3. **Negative (non-object) dataset** - same multi-zip upload + checks,
   then label contents are cleared and yaml files removed from every
   folder, combined into one folder, named and downloaded.
4. **Compare & train dataset** - the object dataset's image count must
   not exceed the non-object dataset's count (error otherwise); if it
   passes, both are merged into one final train/valid/test dataset,
   named and downloaded.
