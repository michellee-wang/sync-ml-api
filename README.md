# Sync API : Using LSTM NN to generate music

**HackIllinois** Backend for [**Sync**](https://github.com/michellee-wang/sync)

- This service generates the background music for Sync and drives the in-game terrain
- The model can be fine-tuned or conditioned on Spotify data

Run locally or deploy to [Modal](https://modal.com) and use the web UI to generate and download beats.

## Requirements

- **Python 3.11**

## Quick Start (Local)

```bash
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Generate a track

```bash
python generate.py --weights path/to/checkpoint.hdf5
```
