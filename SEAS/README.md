# SEAS

Official code for SEAS: Semantic Evidence Allocation in Spectral Time-Series Forecasting**.

SEAS studies weakly aligned multimodal forecasting, where retrieved text is not always clean supervision. Instead of globally fusing text and time series, SEAS treats text as uncertain semantic evidence and allocates it to different spectral structures of the historical series, such as low-frequency trend, mid-frequency periodicity, and high-frequency shocks.

## Main Idea

- Map the historical series into the frequency domain.
- Decompose the spectrum into structure-aware bands.
- Distill retrieved text into compact semantic evidence.
- Use semantic evidence to selectively modulate different spectral bands for forecasting.

## Repository Structure

- `models/SEAS.py`: core SEAS model.
- `run.py`: main training and evaluation entry.
- `exp/`: experiment runners.
- `data_provider/`: dataset loading.
- `scripts/`: data preparation and experiment scripts.


## Notes

- Put pretrained language models under `./pretrained/` if needed.
- Large data, outputs, and checkpoints are ignored by default to keep the repository lightweight.
