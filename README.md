# Sync API 
## Using LSTM NN to generate music

**HackIllinois · Modal Inference Track.** Backend for [**Sync**](https://github.com/michellee-wang/sync), the frontend and blockchain skill-based game. This service generates the **background music** for Sync and drives the **in-game terrain** — obstacles and level layout are influenced by the generated MIDI (e.g. beats and intensity). The model can be **fine-tuned or conditioned on Spotify data** (listening history, top tracks, genre) so the soundtrack and terrain adapt to each player’s taste.

Generate an edm/hiphop MIDI with an LSTM neural network. Run locally or deploy to [Modal](https://modal.com) and use the web UI to generate and download beats.

## Features

- **Local:** Train on your MIDI corpus and generate tracks from the command line.
- **Cloud:** Deploy to Modal for a web API and frontend — generate MIDI in the browser, optional GPU training.

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

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `output.mid` | Output MIDI path |
| `--length` | `200` | Number of notes |
| `--temperature` | `0.8` | Sampling (0=safe, 1=natural, 1.5=wild) |

### Train

```bash
python train.py --epochs 200
python train.py --weights checkpoint.hdf5 --epochs 100   # resume
```

## Deploy to Modal (Web API + Frontend)

1. **Auth:** [modal.com](https://modal.com) → sign up, then:
   ```bash
   modal token set
   ```

2. **Upload data and weights** (once):
   ```bash
   python upload_to_modal.py
   ```

3. **Deploy:**
   ```bash
   modal deploy modal_app.py
   ```
   Copy the URL it prints (e.g. `https://YOUR_WORKSPACE--lofi-generator-api.modal.run`).

4. **Frontend:** Open `frontend/index.html` in a browser, paste your API URL, click **Test Connection**, then **Generate** and download MIDI.

## API (after deploy)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/list_checkpoints` | List available model weights |
| POST | `/generate` | Generate MIDI (body: `weights`, `length`, `temperature`) |
| POST | `/train_endpoint` | Start GPU training (body: `epochs`, `batch_size`, `weights?`) |

## Project Structure

```
├── train.py              # Local training
├── generate.py           # Local generation
├── modal_app.py          # Modal deployment (single FastAPI app)
├── upload_to_modal.py    # Upload data/weights to Modal volume
├── frontend/
│   ├── index.html        # Web UI (generate + train)
│   └── game.html         # Lofi Dash game
├── lofi/
│   ├── data.py           # MIDI parsing & sequences
│   ├── network.py        # LSTM model
│   └── midi_io.py        # Token → MIDI
├── midi_songs/            # Training MIDI corpus (add your own)
├── data/notes             # Cached token stream (from prep)
└── checkpoints/           # Saved weights (gitignored)
```

## Dataset

Add MIDI files to `midi_songs/`. The repo references a corpus of lo-fi and jazz-style MIDI; you can use your own. Run the data prep (see `lofi/data.py` / training flow) to produce `data/notes` for training.

this dataset is from kaggle# sync-ml-api
# sync-ml-api
# sync-ml-api
# sync-ml-api
# sync-ml-api
# sync-ml-api
