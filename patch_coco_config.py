import json

path = "notebooks/coco_cnn_anfis.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Patch cell konfigurasi (cari berdasarkan konten OUTPUT_DIR)
NEW_CONFIG_SOURCE = """\
from pathlib import Path

SEED = 42

# ── Path ────────────────────────────────────────────────────────────────────
# Di Kaggle semua output WAJIB di /kaggle/working (satu-satunya folder writable)
# Di lokal otomatis pakai path relatif dari project root
if Path("/kaggle").exists():
    IMAGE_DIR       = Path("/kaggle/input/datasets/dzikribassyril/dronewate-data/data/raw/images")
    ANNOTATION_JSON = Path("/kaggle/input/datasets/dzikribassyril/dronewate-data/data/raw/annotations/dronewaste_v2.0.json")
    OUTPUT_DIR      = Path("/kaggle/working/models/coco_mask_cnn_anfis")
else:
    IMAGE_DIR       = Path("data/raw/images")
    ANNOTATION_JSON = Path("data/raw/annotations/dronewaste_v2.0.json")
    OUTPUT_DIR      = Path("models/coco_mask_cnn_anfis")

# ── Grid & Label ────────────────────────────────────────────────────────────
GRID_SIZE                  = 128
SAFE_DENSITY_MAX           = 0.05
CRITICAL_DENSITY_MIN       = 0.40
COCO_CANDIDATE_DENSITY_MIN = 0.0   # > 0 berarti grid punya piksel mask sampah

# ── Model ───────────────────────────────────────────────────────────────────
CNN_BATCH_SIZE    = 128
ANFIS_BATCH_SIZE  = 512
NUM_WORKERS       = 0

PCA_COMPONENTS     = 24
ANFIS_RULES        = 12
ANFIS_EPOCHS       = 120
ANFIS_LR           = 0.003
ANFIS_WEIGHT_DECAY = 1e-5

SAFE_SAMPLE_RATIO = 1.0   # Jumlah grid Aman murni relatif terhadap grid kandidat mask

# ── Device & Seed ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Buat folder output (mkdir aman karena sudah Path object)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Device    :", DEVICE)
print("Output dir:", OUTPUT_DIR, "| exists:", OUTPUT_DIR.exists())
"""

patched = False
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "OUTPUT_DIR" in src and ("SAFE_DENSITY_MAX" in src or "ANFIS_RULES" in src):
        cell["source"] = NEW_CONFIG_SOURCE
        patched = True
        print(f"  Patched config cell (id={cell
