import os
import shutil
import threading
import time
import uuid
import zipfile
from typing import List

import yaml
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(BASE, "workspace")
os.makedirs(WORK_DIR, exist_ok=True)

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FastAPI(title="YOLO Dataset Trainer")
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE, "templates"))

# batch_id -> { kind, dir, parts: {part_id: {filename, root}}, final, zip, folder_name }
DB = {}
# session_id -> { dir, classes: {class_name: {dir, zip, count}} }
SEP_DB = {}

# One lock per batch_id so a double-click / duplicate request can never run
# combine (or finalize) twice at once on the same folder.
_LOCKS_GUARD = threading.Lock()
_BATCH_LOCKS = {}


def get_lock(batch_id):
    with _LOCKS_GUARD:
        if batch_id not in _BATCH_LOCKS:
            _BATCH_LOCKS[batch_id] = threading.Lock()
        return _BATCH_LOCKS[batch_id]


def err(message, status=400):
    return JSONResponse({"error": message}, status_code=status)


def safe_rmtree(path, attempts=6, delay=0.25):
    """shutil.rmtree, but retries briefly if a file is still momentarily
    locked (antivirus / indexer) instead of crashing the request."""
    if not os.path.isdir(path):
        return
    last_err = None
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as e:
            last_err = e
            time.sleep(delay)
    raise last_err


# --------------------------------------------------------------- helpers
def find_root(extract_path):
    """Zip files are often wrapped in one or two extra folders - drill down
    until we find the level that actually contains train/valid/test."""
    entries = [e for e in os.listdir(extract_path) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_path, entries[0])):
        root = os.path.join(extract_path, entries[0])
        inner = [e for e in os.listdir(root) if not e.startswith(".")]
        if len(inner) == 1 and os.path.isdir(os.path.join(root, inner[0])):
            sub = os.path.join(root, inner[0])
            if all(os.path.isdir(os.path.join(sub, s)) for s in SPLITS):
                return sub
        return root
    return extract_path


def list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXTS)


def list_labels(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".txt"))


def structure_errors(root):
    errors = []
    if not os.path.isfile(os.path.join(root, "data.yaml")):
        errors.append("data.yaml is missing in dataset root")
    for s in SPLITS:
        d = os.path.join(root, s)
        if not os.path.isdir(d):
            errors.append(f"missing folder: {s}/")
            continue
        if not os.path.isdir(os.path.join(d, "images")):
            errors.append(f"missing folder: {s}/images")
        if not os.path.isdir(os.path.join(d, "labels")):
            errors.append(f"missing folder: {s}/labels")
    return errors


def split_details(root):
    details, problems = {}, []
    for s in SPLITS:
        idir, ldir = os.path.join(root, s, "images"), os.path.join(root, s, "labels")
        imgs, lbls = list_images(idir), list_labels(ldir)
        img_stems = {os.path.splitext(f)[0] for f in imgs}
        lbl_stems = {os.path.splitext(f)[0] for f in lbls}
        orphan_img = sorted(img_stems - lbl_stems)
        orphan_lbl = sorted(lbl_stems - img_stems)
        empty = 0
        if os.path.isdir(ldir):
            empty = sum(1 for f in lbls if os.path.getsize(os.path.join(ldir, f)) == 0)
        details[s] = {"images": len(imgs), "labels": len(lbls),
                      "missing_labels": len(orphan_img),
                      "missing_images": len(orphan_lbl),
                      "empty_labels": empty}
        if orphan_img:
            pv = ", ".join(orphan_img[:5]) + (" ..." if len(orphan_img) > 5 else "")
            problems.append(f"{s}: {len(orphan_img)} image(s) WITHOUT label -> {pv}")
        if orphan_lbl:
            pv = ", ".join(orphan_lbl[:5]) + (" ..." if len(orphan_lbl) > 5 else "")
            problems.append(f"{s}: {len(orphan_lbl)} label(s) WITHOUT image -> {pv}")
    return details, problems


def build_report(title, details, errors, warnings):
    lines = [title, "-" * max(len(title), 12)]
    for s in SPLITS:
        d = details.get(s, {})
        lines.append(s.upper())
        lines.append(f"  Images         : {d.get('images', 0)}")
        lines.append(f"  Labels         : {d.get('labels', 0)}")
        lines.append(f"  Missing Labels : {d.get('missing_labels', 0)}")
        lines.append(f"  Empty Labels   : {d.get('empty_labels', 0)}")
    ti = sum(d.get("images", 0) for d in details.values())
    tl = sum(d.get("labels", 0) for d in details.values())
    lines.append(f"TOTAL : Images {ti} | Labels {tl}")
    lines += ["", "ERRORS", "------"]
    lines += errors if errors else ["No errors found"]
    lines += ["", "WARNINGS", "--------"]
    lines += warnings if warnings else ["No warnings"]
    return "\n".join(lines)


def merge_totals(details_list):
    agg = {s: {"images": 0, "labels": 0, "missing_labels": 0,
               "missing_images": 0, "empty_labels": 0} for s in SPLITS}
    for details in details_list:
        for s in SPLITS:
            d = details.get(s, {})
            for k in agg[s]:
                agg[s][k] += d.get(k, 0)
    return agg


def read_yaml_at(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def write_yaml_at(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def yaml_names(root):
    data = read_yaml_at(os.path.join(root, "data.yaml"))
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return []


def zip_directory(folder, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for base, _dirs, files in os.walk(folder):
            for fn in files:
                fp = os.path.join(base, fn)
                zf.write(fp, os.path.relpath(fp, folder))


def sanitize_name(name):
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name).strip())
    safe = safe.strip("_")
    return safe or "class"


async def save_upload(upload: UploadFile, dst_path: str):
    content = await upload.read()
    with open(dst_path, "wb") as out:
        out.write(content)


KIND_MAP = {"pos": "object", "neg": "nonobject",
            "positive": "object", "negative": "nonobject",
            "object": "object", "nonobject": "nonobject"}


# --------------------------------------------------------------- page
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# =================================================================
# STEP 1 — class separator: split one uploaded dataset by class name
# =================================================================
def class_name_for(idx, names):
    if 0 <= idx < len(names):
        return names[idx]
    return f"class_{idx}"


def separate_by_class(root, out_dir, names):
    """
    Separate one YOLO dataset by class.

    Output format for each class:

    class_name/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml

    Each output class dataset contains only that class.
    The class index inside label files is rewritten to 0.
    """

    class_dirs = {}
    counts = {}

    # -----------------------------------------------------------
    # Create class dataset folders
    # -----------------------------------------------------------
    for idx, class_name in enumerate(names):
        cname = sanitize_name(class_name)

        cdir = os.path.join(out_dir, cname)

        for split in SPLITS:
            os.makedirs(
                os.path.join(cdir, split, "images"),
                exist_ok=True
            )
            os.makedirs(
                os.path.join(cdir, split, "labels"),
                exist_ok=True
            )

        # Create class-specific data.yaml
        yaml_path = os.path.join(cdir, "data.yaml")

        yaml_data = {
            "path": ".",
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 1,
            "names": [class_name]
        }

        write_yaml_at(yaml_path, yaml_data)

        class_dirs[cname] = cdir
        counts[cname] = 0

    # -----------------------------------------------------------
    # Process train / valid / test
    # -----------------------------------------------------------
    for split in SPLITS:

        idir = os.path.join(root, split, "images")
        ldir = os.path.join(root, split, "labels")

        if not os.path.isdir(idir):
            continue

        for img in list_images(idir):

            stem = os.path.splitext(img)[0]
            lbl_path = os.path.join(ldir, stem + ".txt")

            # No label -> skip
            if not os.path.isfile(lbl_path):
                continue

            # ---------------------------------------------------
            # Read YOLO label
            # ---------------------------------------------------
            with open(
                lbl_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                lines = [
                    ln.strip()
                    for ln in f
                    if ln.strip()
                ]

            # ---------------------------------------------------
            # Group annotations by original class index
            # ---------------------------------------------------
            by_class = {}

            for line in lines:

                parts = line.split()

                if not parts:
                    continue

                try:
                    idx = int(float(parts[0]))
                except ValueError:
                    continue

                # Unknown class index
                if idx < 0 or idx >= len(names):
                    continue

                # Rewrite class index to 0
                parts[0] = "0"

                new_line = " ".join(parts)

                by_class.setdefault(idx, []).append(new_line)

            # ---------------------------------------------------
            # Create image + label for every class present
            # ---------------------------------------------------
            for idx, new_lines in by_class.items():

                original_class_name = class_name_for(idx, names)
                cname = sanitize_name(original_class_name)

                if cname not in class_dirs:
                    continue

                cdir = class_dirs[cname]

                dst_img_dir = os.path.join(
                    cdir,
                    split,
                    "images"
                )

                dst_lbl_dir = os.path.join(
                    cdir,
                    split,
                    "labels"
                )

                # ------------------------------------------------
                # Avoid duplicate filename
                # ------------------------------------------------
                base_stem, ext = os.path.splitext(img)

                dst_name = img
                n = 1

                while os.path.exists(
                    os.path.join(dst_img_dir, dst_name)
                ):
                    dst_name = f"{base_stem}_{n}{ext}"
                    n += 1

                # ------------------------------------------------
                # Copy image
                # ------------------------------------------------
                shutil.copy2(
                    os.path.join(idir, img),
                    os.path.join(dst_img_dir, dst_name)
                )

                # ------------------------------------------------
                # Create class-specific label
                # ------------------------------------------------
                dst_label_name = (
                    os.path.splitext(dst_name)[0] + ".txt"
                )

                dst_label_path = os.path.join(
                    dst_lbl_dir,
                    dst_label_name
                )

                with open(
                    dst_label_path,
                    "w",
                    encoding="utf-8"
                ) as f:
                    f.write(
                        "\n".join(new_lines) + "\n"
                    )

                counts[cname] += 1

    return class_dirs, counts


@app.post("/api/separate/upload")
async def separate_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        return err("please upload a .zip dataset")

    session_id = uuid.uuid4().hex[:10]
    sdir = os.path.join(WORK_DIR, "separate", session_id)
    os.makedirs(sdir, exist_ok=True)
    zpath = os.path.join(sdir, "upload.zip")
    await save_upload(file, zpath)

    extract = os.path.join(sdir, "extracted")
    try:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(extract)
    except zipfile.BadZipFile:
        shutil.rmtree(sdir, ignore_errors=True)
        return err("invalid zip file")

    root = find_root(extract)
    errors = structure_errors(root)
    if errors:
        return {"ok": False, "errors": errors,
                "report": build_report("STRUCTURE CHECK", {}, errors, [])}

    names = yaml_names(root)
    out_dir = os.path.join(sdir, "classes")
    class_dirs, counts = separate_by_class(root, out_dir, names)
    if not class_dirs:
        return err("no annotated objects were found to separate")

    classes = {}
    for cname, cdir in class_dirs.items():
        zpath2 = os.path.join(sdir, f"{cname}.zip")
        zip_directory(cdir, zpath2)
        classes[cname] = {"dir": cdir, "zip": zpath2, "count": counts[cname]}
    SEP_DB[session_id] = {"dir": sdir, "classes": classes}

    return {"ok": True, "session_id": session_id,
            "classes": [{"name": c, "count": classes[c]["count"]} for c in sorted(classes)]}


@app.get("/api/separate/download/{session_id}/{class_name}")
def separate_download(session_id: str, class_name: str):
    sess = SEP_DB.get(session_id)
    if not sess or class_name not in sess["classes"]:
        return err("class folder not found", 404)
    zpath = sess["classes"][class_name]["zip"]
    return FileResponse(zpath, filename=os.path.basename(zpath), media_type="application/zip")


# =================================================================
# STEP 3/7 — upload MULTIPLE zip folders for object / non-object dataset
# =================================================================
@app.post("/api/upload/{kind}")
async def upload(kind: str, files: List[UploadFile] = File(...)):
    kind = KIND_MAP.get(kind.lower(), kind)
    if kind not in ("object", "nonobject"):
        return err("unknown dataset kind")

    files = [f for f in files if f and f.filename]
    if not files:
        return err("please choose at least one .zip dataset")
    bad = [f.filename for f in files if not f.filename.lower().endswith(".zip")]
    if bad:
        return err(f"only .zip files are accepted: {', '.join(bad)}")

    batch_id = uuid.uuid4().hex[:10]
    batch_dir = os.path.join(WORK_DIR, batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    parts = {}
    for f in files:
        part_id = uuid.uuid4().hex[:8]
        part_dir = os.path.join(batch_dir, "parts", part_id)
        os.makedirs(part_dir, exist_ok=True)
        zpath = os.path.join(part_dir, "upload.zip")
        await save_upload(f, zpath)
        extract = os.path.join(part_dir, "extracted")
        try:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(extract)
        except zipfile.BadZipFile:
            shutil.rmtree(batch_dir, ignore_errors=True)
            return err(f"invalid zip file: {f.filename}")
        parts[part_id] = {"filename": f.filename, "root": find_root(extract)}

    DB[batch_id] = {"kind": kind, "dir": batch_dir, "parts": parts,
                     "final": None, "zip": None, "folder_name": None}
    return {"batch_id": batch_id,
            "files": [{"part_id": pid, "filename": p["filename"]} for pid, p in parts.items()]}


# --------------------------------------------------------------- check (per zip + combined)
@app.get("/api/check/{batch_id}")
def check(batch_id: str):
    ds = DB.get(batch_id)
    if not ds:
        return err("dataset not found - upload again", 404)

    per_file, all_details = [], []
    for part in ds["parts"].values():
        root = part["root"]
        errors = structure_errors(root)
        details, problems, warnings = {}, [], []
        if not errors:
            details, problems = split_details(root)
            if ds["kind"] == "object":
                empty_total = sum(d.get("empty_labels", 0) for d in details.values())
                if empty_total:
                    warnings.append(f"{empty_total} empty label file(s) found")
        file_errors = errors + problems
        all_details.append(details)
        per_file.append({
            "filename": part["filename"],
            "ok": len(file_errors) == 0,
            "errors": file_errors,
            "warnings": warnings,
            "details": details,
            "report": build_report(part["filename"], details, file_errors, warnings),
        })

    ok = all(pf["ok"] for pf in per_file)
    agg_details = merge_totals(all_details) if ok else {}
    agg_errors = [] if ok else ["fix the errors shown for each file below before continuing"]
    return {"ok": ok, "files": per_file,
            "aggregate": {"details": agg_details,
                          "report": build_report("COMBINED TOTAL (all files)", agg_details, agg_errors, [])}}


# --------------------------------------------------------------- class name check/change (all zips)
class ClassBody(BaseModel):
    batch_id: str
    class_name: str


@app.post("/api/class/check")
def class_check(body: ClassBody):
    ds = DB.get(body.batch_id)
    cls = body.class_name.strip()
    if not ds:
        return err("dataset not found", 404)
    if not cls:
        return err("enter the class name")

    per_file, all_matched = [], True
    for part in ds["parts"].values():
        names = yaml_names(part["root"])
        matched = bool(names) and cls.lower() in [n.lower() for n in names]
        all_matched = all_matched and matched
        per_file.append({"filename": part["filename"], "yaml_names": names, "matched": matched})
    return {"matched": all_matched, "class_name": cls, "files": per_file}


@app.post("/api/class/change")
def class_change(body: ClassBody):
    ds = DB.get(body.batch_id)
    cls = body.class_name.strip()
    if not ds or not cls:
        return err("dataset id and class name required")
    changed = []
    for part in ds["parts"].values():
        root = part["root"]
        targets = [os.path.join(root, "data.yaml")]
        for s in SPLITS:
            targets.append(os.path.join(root, s, "data.yaml"))
        for p in targets:
            if os.path.isfile(p):
                data = read_yaml_at(p)
                data["names"] = [cls]
                data["nc"] = 1
                write_yaml_at(p, data)
                changed.append(f"{part['filename']} -> {os.path.relpath(p, root)}")
    return {"ok": bool(changed), "changed": changed, "class_name": cls}


# --------------------------------------------------------------- 7:3 non-object cleaning (all zips)
class BatchBody(BaseModel):
    batch_id: str


@app.post("/api/nonobject/clean")
def clean_nonobject(body: BatchBody):
    ds = DB.get(body.batch_id)
    if not ds or ds["kind"] != "nonobject":
        return err("non-object dataset not found", 404)
    emptied = created = removed_yaml = 0
    for part in ds["parts"].values():
        root = part["root"]
        for s in SPLITS:
            idir, ldir = os.path.join(root, s, "images"), os.path.join(root, s, "labels")
            if not os.path.isdir(ldir):
                os.makedirs(ldir, exist_ok=True)
            lbls = set(list_labels(ldir))
            for f in list_images(idir):
                stem = os.path.splitext(f)[0] + ".txt"
                p = os.path.join(ldir, stem)
                if stem not in lbls:
                    open(p, "w").close()
                    created += 1
                elif os.path.getsize(p) > 0:
                    open(p, "w").close()
                    emptied += 1
        for base, _dirs, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith((".yaml", ".yml")):
                    os.remove(os.path.join(base, fn))
                    removed_yaml += 1
    return {"ok": True, "emptied_labels": emptied, "created_labels": created, "removed_yaml": removed_yaml}


# --------------------------------------------------------------- 5/8 combine ALL zips into ONE folder
@app.post("/api/combine")
def combine(body: BatchBody):
    batch_id = body.batch_id
    ds = DB.get(batch_id)
    if not ds:
        return err("dataset not found", 404)

    lock = get_lock(batch_id)
    if not lock.acquire(blocking=False):
        return err("this dataset is already being combined - please wait a moment and try again", 409)

    try:
        final = os.path.join(ds["dir"], "final")
        safe_rmtree(final)
        for s in SPLITS:
            os.makedirs(os.path.join(final, s, "images"), exist_ok=True)
            os.makedirs(os.path.join(final, s, "labels"), exist_ok=True)

        counts = {s: {"images": 0, "labels": 0} for s in SPLITS}
        names = []
        for part in ds["parts"].values():
            root = part["root"]
            if ds["kind"] == "object" and not names:
                names = yaml_names(root)
            for s in SPLITS:
                idir, ldir = os.path.join(root, s, "images"), os.path.join(root, s, "labels")
                for f in list_images(idir):
                    stem, ext = os.path.splitext(f)
                    dst_name = f
                    while os.path.exists(os.path.join(final, s, "images", dst_name)):
                        dst_name = f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
                    shutil.copy2(os.path.join(idir, f), os.path.join(final, s, "images", dst_name))
                    dst_stem = os.path.splitext(dst_name)[0]
                    src_lbl = os.path.join(ldir, stem + ".txt")
                    dst_lbl = os.path.join(final, s, "labels", dst_stem + ".txt")
                    if os.path.isfile(src_lbl):
                        shutil.copy2(src_lbl, dst_lbl)
                    elif ds["kind"] == "nonobject":
                        open(dst_lbl, "w").close()
                    counts[s]["images"] += 1
        for s in SPLITS:
            counts[s]["labels"] = len(list_labels(os.path.join(final, s, "labels")))

        if ds["kind"] == "object":
            write_yaml_at(os.path.join(final, "data.yaml"), {
                "path": ".", "train": "train/images", "val": "valid/images",
                "test": "test/images", "nc": len(names) or 1,
                "names": names or ["object"]})

        ds["final"] = final
        totals = {"images": sum(c["images"] for c in counts.values()),
                  "labels": sum(c["labels"] for c in counts.values())}
        return {"ok": True, "counts": counts, "totals": totals, "names": names,
                "source_files": len(ds["parts"])}
    finally:
        lock.release()


# --------------------------------------------------------------- 6/9/final — finalize + download
class FinalizeBody(BaseModel):
    batch_id: str
    folder_name: str


@app.post("/api/finalize")
def finalize(body: FinalizeBody):
    batch_id = body.batch_id
    ds = DB.get(batch_id)
    folder = body.folder_name.strip()
    if not ds or not ds.get("final"):
        return err("combine the dataset first")
    if not folder:
        return err("enter the folder name")

    lock = get_lock(batch_id)
    if not lock.acquire(blocking=False):
        return err("this dataset is busy - please wait a moment and try again", 409)
    try:
        safe = sanitize_name(folder)
        zpath = os.path.join(ds["dir"], safe + ".zip")
        if os.path.exists(zpath):
            os.remove(zpath)
        zip_directory(ds["final"], zpath)
        ds["zip"], ds["folder_name"] = zpath, safe
        return {"ok": True, "folder_name": safe, "download_url": f"/api/download/{batch_id}"}
    finally:
        lock.release()


@app.get("/api/download/{batch_id}")
def download(batch_id: str):
    ds = DB.get(batch_id)
    if not ds or not ds.get("zip") or not os.path.isfile(ds["zip"]):
        return err("nothing to download yet", 404)
    return FileResponse(ds["zip"], filename=os.path.basename(ds["zip"]), media_type="application/zip")


# =================================================================
# STEP 9 — compare object vs non-object counts, then build the train dataset
# object_count must NOT exceed nonobject_count; otherwise it's an error.
# =================================================================
class BalanceBody(BaseModel):
    object_batch_id: str
    nonobject_batch_id: str


@app.post("/api/balance/check")
def balance_check(body: BalanceBody):
    pos, neg = DB.get(body.object_batch_id), DB.get(body.nonobject_batch_id)
    if not pos or not neg or not pos.get("final") or not neg.get("final"):
        return err("finalize both datasets first")

    def total(ds):
        return sum(len(list_images(os.path.join(ds["final"], s, "images"))) for s in SPLITS)

    pc, nc = total(pos), total(neg)
    balanced = pc >= nc  # object count must be <= non-object count
    return {"ok": True, "object_count": pc, "nonobject_count": nc, "balanced": balanced}


@app.post("/api/train/combine")
def train_combine(body: BalanceBody):
    pos, neg = DB.get(body.object_batch_id), DB.get(body.nonobject_batch_id)
    if not pos or not neg or not pos.get("final") or not neg.get("final"):
        return err("finalize both datasets first")

    pc = sum(len(list_images(os.path.join(pos["final"], s, "images"))) for s in SPLITS)
    nc = sum(len(list_images(os.path.join(neg["final"], s, "images"))) for s in SPLITS)
    if pc < nc:
      return err(
            f"Object detection dataset ({pc} images) is less than the "
            f"non-object dataset ({nc} images). "
            f"Object detection images must be greater than or equal to "
            f"non-object images before combining."
        )

    out = os.path.join(WORK_DIR, "train_" + uuid.uuid4().hex[:8])
    counts = {}
    for s in SPLITS:
        os.makedirs(os.path.join(out, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, s, "labels"), exist_ok=True)
        n = 0
        for src in (pos["final"], neg["final"]):
            idir, ldir = os.path.join(src, s, "images"), os.path.join(src, s, "labels")
            for f in list_images(idir):
                stem, ext = os.path.splitext(f)
                dst = f
                while os.path.exists(os.path.join(out, s, "images", dst)):
                    dst = f"{stem}_{uuid.uuid4().hex[:4]}{ext}"
                shutil.copy2(os.path.join(idir, f), os.path.join(out, s, "images", dst))
                lbl_src = os.path.join(ldir, stem + ".txt")
                lbl_dst = os.path.splitext(dst)[0] + ".txt"
                if os.path.isfile(lbl_src):
                    shutil.copy2(lbl_src, os.path.join(out, s, "labels", lbl_dst))
                else:
                    open(os.path.join(out, s, "labels", lbl_dst), "w").close()
                n += 1
        counts[s] = {"images": n, "labels": len(list_labels(os.path.join(out, s, "labels")))}

    names = yaml_names(pos["final"])
    write_yaml_at(os.path.join(out, "data.yaml"), {
        "path": ".", "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": len(names) or 1,
        "names": names or ["object"]})

    tid = "train_" + uuid.uuid4().hex[:8]
    DB[tid] = {"kind": "train", "dir": out, "parts": {}, "final": out, "zip": None, "folder_name": None}
    totals = {"images": sum(c["images"] for c in counts.values()),
              "labels": sum(c["labels"] for c in counts.values())}
    return {"ok": True, "batch_id": tid, "counts": counts, "totals": totals, "names": names}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
