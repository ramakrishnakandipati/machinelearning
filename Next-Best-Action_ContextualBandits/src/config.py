"""Central configuration: paths, seeds, device and the discrete action space."""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
PLOTS_DIR = OUTPUT_DIR / "plots"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"

for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, SCREENSHOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_ACTIONS = 6
ACTIONS = {
    0: "Control / No Offer",
    1: "10% Room Rate Discount",
    2: "Complimentary Room Upgrade",
    3: "$50 Dining & Spa Credit",
    4: "Late Checkout + Priority Transfer",
    5: "Double Loyalty Points",
}
ACTION_SHORT = ["Control", "10% Discount", "Room Upgrade", "$50 Dining/Spa", "Late C/O + Transfer", "2x Points"]

# Semantic arm attributes used by the hybrid LinUCB (shared coefficients across arms):
# [price_cut, room_perk, ancillary_perk, convenience, loyalty_perk, any_offer]
ACTION_ATTR = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0.3, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0.4, 0, 1, 0, 1],
        [0, 0, 0.2, 0, 1, 1],
    ],
    dtype=np.float64,
)

# Categorical vocabularies (fixed so the API and the training pipeline agree)
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
CHANNELS = ["Web", "App", "OTA"]
ARCHETYPES = ["Business", "Family", "Couple", "Solo"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Contribution-margin economics (shared by generator and docs)
ROOM_MARGIN = 0.65
ANC_MARGIN = 0.45
