"""Generate expected values for the comprehensive dashboard test matrix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from ai_services.dsi_fusion_member4.softmax_gmu import GatedMultimodalFusionEngine
from ai_services.dsi_fusion_member4.config import MODEL_WEIGHTS_PATH, GATE_HIDDEN_DIM
from ai_services.dsi_fusion_member4.discordance_delta import calculate_discordance_delta, classify_masking
from ai_services.dsi_fusion_member4.severity import classify_risk_tier

model = GatedMultimodalFusionEngine(hidden_dim=GATE_HIDDEN_DIM)
if MODEL_WEIGHTS_PATH.exists():
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
model.eval()

test_cases = [
    ("Honest: all mid",          0.30, 0.35, 0.28, "# Moderate, no masking"),
    ("Honest: all equal",        0.50, 0.50, 0.50, "# All same, balanced weights"),
    ("Honest: all low (GREEN)",  0.10, 0.10, 0.10, "# No distress"),
    ("Honest: all mid (GREEN)",  0.35, 0.40, 0.35, "# Low-moderate, no masking"),
    ("Honest: all mid (YELLOW)", 0.50, 0.55, 0.52, "# Moderate distress, no masking"),
    ("Honest: all high (RED)",   0.85, 0.82, 0.88, "# Severe distress, consistent"),
    ("Masking: TBS low, VBS/ABS high (YELLOW)", 0.15, 0.70, 0.65, "# Fake happy text, moderate non-verbal"),
    ("Masking: TBS low, VBS/ABS high (RED)",    0.10, 0.85, 0.80, "# Fake happy text, severe non-verbal"),
    ("Masking: TBS low, VBS high only (YELLOW)", 0.20, 0.75, 0.30, "# Only video is high"),
    ("Forced: TBS high, VBS/ABS low (GREEN)",   0.92, 0.08, 0.07, "# YOUR EXACT INPUT!"),
    ("Forced: TBS high, VBS/ABS low (GREEN)",   0.85, 0.20, 0.15, "# Mild composure"),
    ("Forced: TBS high, VBS/ABS low (YELLOW)",  0.90, 0.35, 0.30, "# High text, moderate non-verbal"),
    ("Forced: TBS extreme, VBS/ABS extreme (GRE)", 0.96, 0.05, 0.05, "# Extreme test case"),
    ("Boundary: delta exactly +0.50", 0.30, 0.80, 0.30, "# delta=+0.50, NOT masking (strict >)"),
    ("Boundary: delta just above +0.50", 0.29, 0.80, 0.30, "# delta=+0.51, masking triggers"),
    ("Boundary: delta exactly -0.50", 0.80, 0.30, 0.30, "# delta=-0.50, NOT masking (strict >)"),
    ("Boundary: delta just below -0.50", 0.81, 0.30, 0.30, "# delta=-0.51, masking triggers"),
    ("Boundary: DSI just above 0.45 (YELLOW)", 0.451, 0.451, 0.451, "# Just over GREEN threshold"),
    ("Boundary: DSI just below 0.70 (YELLOW)", 0.699, 0.699, 0.699, "# Just under RED threshold"),
    ("Edge: all zeros", 0.0, 0.0, 0.0, "# Everything zero, DSI=0.00"),
    ("Edge: all ones", 1.0, 1.0, 1.0, "# Everything max, DSI=1.00"),
    ("Edge: TBS=1.0,VBS=ABS=0 (GRE)", 1.0, 0.0, 0.0, "# Extreme forced composure"),
    ("Edge: TBS=0.0,VBS=ABS=1.0 (RED)", 0.0, 1.0, 1.0, "# Extreme traditional masking"),
    ("Mixed: moderate discord (YELLOW)", 0.55, 0.30, 0.40, "# Some discordance, moderate DSI"),
    ("Mixed: TBS>VBS>ABS", 0.70, 0.50, 0.40, "# Staircase descending, no masking"),
    ("Mixed: TBS<VBS<ABS", 0.30, 0.50, 0.70, "# Staircase ascending, no masking"),
    ("Mixed: VBS dominant (YELLOW)", 0.45, 0.65, 0.50, "# Video clearly highest, no masking"),
    ("Discord: TBS>VBS=ABS (small)", 0.55, 0.30, 0.30, "# delta=-0.25, small forced composure"),
    ("Discord: TBS<VBS=ABS (small)", 0.30, 0.55, 0.55, "# delta=+0.25, small traditional masking"),
]

print("=" * 125)
print(f"{'TEST CASE':<45} {'TBS':>5} {'VBS':>5} {'ABS':>5} {'delta':>7} {'DSI':>7} {'w_T':>5} {'w_V':>5} {'w_A':>5} {'TIER':>8} {'MASKING':<14}")
print("=" * 125)

for label, tbs, vbs, abs_, desc in test_cases:
    delta = calculate_discordance_delta(tbs, vbs, abs_)
    mask = classify_masking(delta)
    dsi, weights, _ = model.predict(tbs, vbs, abs_)
    tier = classify_risk_tier(dsi)
    wt, wv, wa = weights[0], weights[1], weights[2]
    masking_str = mask["masking_type"] if mask["is_masking"] else "-"
    print(f"{label:<45} {tbs:>5.3f} {vbs:>5.3f} {abs_:>5.3f} {delta:>+7.2f} {dsi:>7.4f} {wt:>5.2f} {wv:>5.2f} {wa:>5.2f} {tier.name:<8} {masking_str:<14}")

print("=" * 125)

# Also print a human-readable summary
print()
print("SCENARIO LEGEND:")
print("  delta > +0.5  -> Traditional Masking (fake-happy text, distressed exterior)")
print("  delta < -0.5  -> Forced Composure (positive text, distressed exterior)")
print("  |delta| <= 0.5 -> No masking (consistent modalities)")
print("  DSI < 0.45    -> GREEN tier (low risk)")
print("  0.45 <= DSI < 0.70 -> YELLOW tier (moderate risk)")
print("  DSI >= 0.70   -> RED tier (high risk)")
