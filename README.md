# MindSync-Enterprise — Multimodal Depression Detection

Central Decision Engine (DSI fusion) & HR Analytics Dashboard. Fuses text, video and audio
scores into a Depression Severity Index (DSI) with emotional-masking detection.

## Overview

This component receives 3 normalized float scores per check-in:
- **TBS** (Text-Based Score, float 0.0 - 1.0)
- **VBS** (Video-Based Score, float 0.0 - 1.0)
- **ABS** (Audio-Based Score, float 0.0 - 1.0)

It performs **late-stage decision-level fusion** through 3 stages:

### 1. Discordance Delta Calculation
```
Δ = max(VBS, ABS) - TBS
```
- **High positive Δ (> +0.50)**: Traditional Emotional Masking (Fake Happy Text)
- **High negative Δ (< -0.50)**: Forced Composure (Distressed Text, Calm Exterior)

### 2. PyTorch Gated Multimodal Fusion Engine (GMU)
- Ingests `[TBS, VBS, ABS, Δ]` into a gating MLP (`Linear(4, 16) → ReLU → Linear(16, 3)`,
  `hidden_dim=0` falls back to the plain `Linear(4, 3)` baseline)
- Applies `Softmax(dim=-1)` to yield dynamic weights `[w_text, w_video, w_audio]` summing to 1.0
- Calculates the final Depression Severity Index (DSI):
```
DSI = (w_text * TBS) + (w_video * VBS) + (w_audio * ABS)
```

### 3. Risk Tiers & Alerts
| DSI Range | Risk Tier | Status |
|-----------|-----------|--------|
| DSI >= 0.70 | RED TIER | High Risk |
| 0.45 <= DSI < 0.70 | YELLOW TIER | Moderate Risk |
| DSI < 0.45 | GREEN TIER | Low Risk |
## Project Structure

``
MindSync-Enterprise/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── .env.example
│
├── app/                                <-- HR dashboard app (PySide6)
│   ├── main.py                         <-- Desktop UI entry point
│   ├── config.py                       <-- App-level config placeholder
│   ├── resources/styles/main.qss       <-- Legacy stylesheet (theme now generated)
│   ├── ui/
│   │   ├── __init__.py                 <-- Light/dark theme tokens
│   │   ├── main_window.py              <-- (placeholder)
│   │   ├── dialogs/                    <-- login_dialog, consent_dialog (placeholders)
│   │   ├── pages/                      <-- 11 page placeholders
│   │   └── widgets/
│   │       ├── risk_gauge.py           <-- DSI dial
│   │       ├── session_table.py        <-- Audit log table
│   │       └── trend_chart.py          <-- 14-day trend line (matplotlib)
│   ├── services/
│   │   └── dsi_service.py              <-- AlertController (HIGH_RISK / DISCORDANCE / ESCALATION)
│   ├── database/
│   │   └── connection.py               <-- SQLite DatabaseManager (hr_users, checkins)
│   ├── core/                           <-- security/permissions/validators/exceptions (placeholders)
│   └── utils/
│       └── theme.py                    <-- Light/dark theme manager (QSettings persisted)
│
├── ai_services/
│   ├── text_analysis/                  <-- Member 1 (TBS extractor, placeholder)
│   ├── video_facial_emotion_behavior/  <-- Member 2 (VBS extractor, placeholder)
│   ├── audio_analysis/                 <-- Member 3 (ABS extractor, placeholder)
│   └── dsi_fusion_member4/             <-- [YOUR MODULE: DSI fusion]
│       ├── __init__.py
│       ├── config.py                   <-- Thresholds, paths, training hyperparams
│       ├── discordance_delta.py        <-- Discordance Delta logic
│       ├── softmax_gmu.py              <-- GatedMultimodalFusionEngine
│       ├── severity.py                 <-- Risk-tier classification
│       ├── dsi_calculator.py           <-- (placeholder)
│       ├── train_gmu.py                <-- Training script
│       └── gmu_fusion_model.pth        <-- Trained weights
│
├── data/
│   ├── checkin_dataset.csv             <-- Synthetic training set (2000 rows)
│   └── local_audit_logs.db             <-- SQLite audit log (git-tracked demo data)
│
├── models/
│   └── README.md                       <-- (placeholder)
│
├── tests/
│   ├── test_discordance.py             <-- 10 tests: Delta math + masking labels
│   └── test_fusion_model.py            <-- 19 tests: GMU, risk tiers, DB logging
│
├── scripts/
│   ├── initialize_database.py          <-- (placeholder)
│   └── package_windows.ps1             <-- (placeholder)
│
└── docs/                               <-- architecture/setup/integration/user/privacy (placeholders)
``
## Setup and Installation

``bash
# 1. Open the project folder (NOT the '- Copy' duplicate, if one exists)
cd '4 year Final project sadeesh'

# 2. Install runtime + dev dependencies (matplotlib is required for the trend chart)
pip install -r requirements.txt
pip install -r requirements-dev.txt
``

## Training the Model

``bash
# Generate 2,000 synthetic samples and train for 200 epochs (TRAIN_EPOCHS=200)
python -m ai_services.dsi_fusion_member4.train_gmu
``

This will:
1. Generate 2,000 synthetic check-in samples covering 4 scenarios:
   - Honest / Consistent (40 pct = 800)
   - Traditional Emotional Masking (25 pct = 500)
   - Forced Composure (25 pct = 500)
   - Mixed / Moderate (10 pct = 200)
2. Train the GMU model for 200 epochs (Adam, MSE loss, ReduceLROnPlateau, best-checkpoint kept)
3. Save the CSV to data/checkin_dataset.csv and weights to ai_services/dsi_fusion_member4/gmu_fusion_model.pth

## Running the HR Dashboard

``bash
# Launch the PySide6 desktop application
python app/main.py
``

### Login Credentials
- **Username**: admin@company.com
- **Password**: admin123

### Dashboard Features
1. **Login Window** - Validates against hr_users table
2. **Header stats** - Sessions / Avg DSI / High risk / Masking pills + light-dark toggle (persisted)
3. **DSI Gauge** - Segmented Green/Yellow/Red dial with needle + today readout
4. **New Check-in Simulator** - TBS/VBS/ABS sliders + Evaluate Session
5. **Weight Breakdown** - Text / Video / Audio pct bars from the GMU gate
6. **Warning Banner** - Triggers when absolute Delta > 0.50 (masking detected)
7. **Audit Log Table** - Search by employee ID + masking-type filter (All / Traditional / Forced / No masking)
8. **Company Trend Chart** - Daily avg DSI line for past 14 days with risk bands; switches to the searched employee personal line when filtering. Load 14-day demo seeds demo rows so the wave is visible.
## Running Tests

``bash
# Full suite (29 tests: 10 discordance + 19 fusion/risk/DB)
python -m pytest tests/ -v

# Single modules
python -m pytest tests/test_discordance.py -v
python -m pytest tests/test_fusion_model.py -v

# With coverage on the fusion package
python -m pytest tests/ -q --cov=ai_services.dsi_fusion_member4 --cov-report=term-missing
``

Notes:
- Tests use temporary SQLite databases - your real data/local_audit_logs.db is untouched.
- tests/unit, tests/integration, tests/ui exist as empty placeholders; the runnable tests are tests/test_discordance.py and tests/test_fusion_model.py.
- What is covered: Discordance Delta math (honest/masking/forced/max/extremes), masking labels incl. exact-threshold 0.50 = no alert, GMU shapes/softmax/DSI range/predict tuple/weight-shifting/gradient-flow/save-load, risk-tier boundaries (incl. 0.45 = YELLOW, 0.70 = RED), DB save/retrieve, masking-alert storage, login valid/invalid, high-risk filter, summary stats.

## Database Schema

### hr_users
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| username | TEXT | Login email (unique) |
| password | TEXT | Login password (stored in plaintext - not encrypted) |
| full_name | TEXT | Full name |
| role | TEXT | User role |
| created_at | TEXT | Creation timestamp |

### checkins
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| employee_id | TEXT | Employee identifier |
| timestamp | TEXT | Check-in timestamp (ISO) |
| tbs | REAL | Text-Based Score (0-1) |
| vbs | REAL | Video-Based Score (0-1) |
| abs | REAL | Audio-Based Score (0-1) |
| dsi | REAL | Fused DSI (0-1) |
| delta | REAL | Discordance Delta |
| risk_tier | TEXT | GREEN / YELLOW / RED |
| masking_alert | TEXT | Masking alert message (NULL when none) |

Privacy note: audit-log contents are NOT encrypted - anyone with the .db file can read all scores. Password hashing + column encryption are planned, not yet implemented.
