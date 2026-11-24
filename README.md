# Humming Bird

A pitch-controlled game: hum or whistle into the microphone and a bird flies
through gaps in walls. The pitch is estimated in real time with DSP.
CS-419 Digital Signal Processing, Open Ended Lab.

## Setup
python -m venv .venv
.venv\Scripts\activate        (Windows)
pip install -r requirements.txt

## Play
```
python -m app.game                       # microphone, with synthetic fallback
python -m app.game --source synthetic    # demo without a microphone
python -m app.game --source data/hum.wav # loop a WAV file
python -m app.game --device 1           # choose an input device
```
Choose Start or press Enter. For a microphone, stay quiet during noise
calibration, then hum the lowest and highest comfortable notes and press Enter
for each. Hum higher to rise and lower to descend through the wall gaps. Brief
pitch dropouts are ignored; longer silence makes the bird fall. Land on a flower
to stop safely, then hum higher to take off. SPACE switches to manual flapping
(and takes off from a flower); P returns to pitch control.

Press D to open the dashboard (gameplay pauses). Its key map shows the live
controls: 1/2/3 select FFT/AMDF/ACF, W changes window, B bypasses the filter,
M switches hum/whistle, [ and ] change frame size, N cycles injected noise, and
C recalibrates the microphone. Escape returns to the game/menu. Synthetic and
WAV sources use a preset pitch range and skip microphone calibration.

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
