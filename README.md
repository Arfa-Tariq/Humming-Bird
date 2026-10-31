# Humming Bird

A pitch-controlled game: hum or whistle into the microphone and a bird flies
through gaps in walls. The pitch is estimated in real time with DSP.
CS-419 Digital Signal Processing, Open Ended Lab.

## Setup
python -m venv .venv
.venv\Scripts\activate        (Windows)
pip install -r requirements.txt

## Run tests
pytest

## Folders
- dsp/          pure DSP code (filters, windows, estimators, post-processing)
- experiments/  scripts that test design choices and save results
- results/      generated figures and CSV files
- app/          game, microphone input and dashboard
- data/         short audio clips only
- report/       final report
- logbook.md    experiment diary

## Rules
- Never commit directly to main. Use a branch.
- dsp/ must never import pygame or sounddevice.
- Only measured results go in the report.

## Team
- Arfa Tariq
- Damiya Naeem
