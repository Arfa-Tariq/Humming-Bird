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

The live dashboard is always visible beside the game; it does not pause play.
Resize the window if needed. Use 1/2/3 for FFT/AMDF/ACF, W for the window,
B to bypass the filter, M for hum/whistle mode, [ and ] for frame size, N for
injected noise, and C to recalibrate the microphone. Escape returns to the menu.
Synthetic and WAV sources use a preset pitch range and skip microphone calibration.

## Reading the dashboard
- **Waveform:** amplitude versus time for one frame. Cyan is the input and orange
  is the filtered signal; compare their shapes to see the filter's effect.
- **Spectrum:** windowed FFT magnitude versus frequency. Peaks show tones and
  harmonics; the white marker is the estimated fundamental (f0), which need not
  be the strongest peak. The displayed dB magnitude is unnormalized, not dBFS,
  and changes with frame size and window.
- **Spectrogram:** frequency versus time, with the newest column on the right.
  Brighter colors mean greater FFT magnitude; bands can show the fundamental and
  harmonics. The horizontal span changes with hop size and available history.
- **Pitch track:** estimated f0 in cyan and smoothed pitch in orange, with a
  logarithmic frequency axis. Missing cyan sections mean no estimate; the orange
  tracker holds its last output, although the bird falls after prolonged silence.
- **Readouts:** an open gate means signal passed the noise threshold, not that
  pitch was detected. Confidence has a method-specific meaning, not a probability.
  The latency budget is an estimate. The shown FFT-bin spacing is fs/N; a larger
  frame generally increases analysis delay while providing a finer frequency grid.
  The window affects the displayed FFT and FFT estimator, not AMDF or ACF.

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
