# Multimodal Depression Detection - Component 4

Central Decision Engine & HR Analytics Dashboard for a Multimodal Depression Detection system.

## Overview

This component receives 3 normalized float scores from a database:
- **TBS** (Text Behavioral Score, float 0.0 - 1.0)
- **VBS** (Video Behavioral Score, float 0.0 - 1.0)
- **ABS** (Audio Behavioral Score, float 0.0 - 1.0)

It performs **Late-Stage Decision-Level Fusion** through 3 stages:

### 1. Discordance Delta Calculation
```
Δ = max(VBS, ABS) - TBS
```
- **High positive Δ (> +0.50)**: Traditional Emotional Masking (Fake Happy Text)
- **High negative Δ (< -0.50)**: Forced Composure (Distressed Text, Calm Exterior)

### 2. PyTorch Gated Multimodal Fusion Engine (GMU)
- Ingests `[TBS, VBS, ABS, Δ]` into an `nn.Linear(4, 3)` gating layer
- Applies `nn.Softmax(dim=-1)` to yield dynamic weights: `[w_text, w_video, w_audio]` summing to 1.0
- Calculates Final Depression Severity Index (DSI):
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

```
multimodal-depression-detection/
├── .gitignore
├── README.md
├── requirements.txt
│
├── core_fusion_engine/            <-- [MAIN MODULE]
│   ├── __init__.py
│   ├── config.py                  <-- App settings and alert thresholds
│   ├── discordance.py             <-- Discordance Delta (Δ) logic
│   ├── fusion_model.py            <-- PyTorch GatedMultimodalFusionEngine
│   ├── gmu_fusion_model.pth       <-- Saved PyTorch model weights
│   ├── severity.py                <-- Risk tier classification
│   └── train_gmu.py               <-- Training script
│
├── hr_dashboard_app/              <-- [UI & CONTROLLER MODULE]
│   ├── __init__.py
│   ├── main_app.py                <-- PySide6 desktop UI entry point
│   ├── database_manager.py        <-- SQLite DB (hr_users, checkins)
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── alert_controller.py    <-- Alert generation logic
│   └── ui_components/
│       ├── __init__.py
│       ├── risk_gauge.py          <-- Green/Yellow/Red DSI gauge
│       └── session_table.py       <-- Audit log table view
│
├── unimodal_services/             <-- [TEAM MEMBERS' MODULES]
│   ├── text_nlp_service/          <-- Member 1 (TBS Extractor)
│   ├── vision_cv_service/         <-- Member 2 (VBS Extractor)
│   └── audio_speech_service/      <-- Member 3 (ABS Extractor)
│
├── data/
│   ├── checkin_dataset.csv        <-- Dataset for model training
│   └── local_audit_logs.db        <-- Local SQLite database
│
└── tests/                         <-- Automated unit tests
    ├── test_discordance.py
    └── test_fusion_model.py
```

## Setup & Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Training the Model

```bash
# Generate 1,000 synthetic samples and train for 40 epochs
python -m core_fusion_engine.train_gmu
```

This will:
1. Generate 1,000 synthetic check-in samples covering 4 scenarios:
   - Honest / Consistent (40%)
   - Traditional Emotional Masking (25%)
   - Forced Composure (25%)
   - Mixed / Moderate (10%)
2. Train the GMU model for 40 epochs using Adam optimizer and MSE loss
3. Save weights to `core_fusion_engine/gmu_fusion_model.pth`

## Running the HR Dashboard

```bash
# Launch the PySide6 desktop application
python hr_dashboard_app/main_app.py
```

### Login Credentials
- **Username**: `admin@company.com`
- **Password**: `admin123`

### Dashboard Features
1. **Login Window** - Validates against `hr_users` table
2. **Simulator Panel** - Manually test TBS, VBS, ABS inputs
3. **DSI Gauge** - Visual Green/Yellow/Red risk indicator
4. **Weight Breakdown** - Shows % weights for Text, Video, Audio
5. **Warning Banner** - Triggers when |Δ| > 0.50 (masking detected)
6. **Audit Log Table** - Displays historical DB records

## Running Tests

```bash
# Run all unit tests
python -m pytest tests/ -v
```

## Database Schema

### hr_users
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| username | TEXT | Login email (unique) |
| password | TEXT | Login password |
| full_name | TEXT | User's full name |
| role | TEXT | User role |
| created_at | TEXT | Creation timestamp |

### checkins
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| employee_id | TEXT | Employee identifier |
| timestamp | TEXT | Check-in timestamp |
| tbs | REAL | Text-Based Score (0-1) |
| vbs | REAL | Video-Based Score (0-1) |
| abs | REAL | Audio-Based Score (0-1) |
| dsi | REAL | Fused DSI (0-1) |
| delta | REAL | Discordance Delta |
| risk_tier | TEXT | GREEN / YELLOW / RED |
| masking_alert | TEXT | Masking alert message |