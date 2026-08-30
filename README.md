# Multimodal Depression Detection

## Project Structure

```
multimodal-depression-detection/
├── .gitignore
├── README.md
├── requirements.txt
│
├── core_fusion_engine/            <-- [YOUR MAIN MODULE]
│   ├── __init__.py
│   ├── config.py                  <-- App settings and alert thresholds
│   ├── fusion_model.py            <-- PyTorch GatedMultimodalFusionEngine architecture
│   ├── gmu_fusion_model.pth       <-- Saved PyTorch model weights
│   ├── train_gmu.py               <-- Script to train/retrain PyTorch model
│   └── discordance.py             <-- Discordance Delta (Δ) logic
│
├── hr_dashboard_app/              <-- [YOUR UI & CONTROLLER MODULE]
│   ├── __init__.py
│   ├── main_app.py                <-- PySide6 desktop UI entry point
│   ├── database_manager.py        <-- DB reader/writer (fetches TBS, VBS, ABS)
│   ├── controllers/               <-- Logic & HR alert controllers
│   │   ├── __init__.py
│   │   └── alert_controller.py
│   └── ui_components/             <-- UI widgets
│       ├── __init__.py
│       ├── risk_gauge.py          <-- Red/Yellow/Green DSI gauge widget
│       └── session_table.py       <-- Audit logs and employee table view
│
├── unimodal_services/             <-- [TEAM MEMBERS' MODULES]
│   ├── text_nlp_service/          <-- Member 1 (TBS Extractor)
│   ├── vision_cv_service/         <-- Member 2 (VBS Extractor)
│   └── audio_speech_service/      <-- Member 3 (ABS Extractor)
│
├── data/
│   ├── checkin_dataset.csv        <-- Dataset for model training
│   └── local_audit_logs.db        <-- Local database file
│
└── tests/                         <-- Automated unit tests
    ├── test_discordance.py
    └── test_fusion_model.py