# Project Log

## December 5, 2025

### Initial Setup
- Explored the repository and understood the project structure.
- Verified that the project is a Visual Speech Recognition (VSR) pipeline aimed at detecting words from a user's lips in real-time.

### Debugging and Fixes
- Identified that the raw VSR output was being logged in uppercase in the terminal.
- Ensured that the LLM-corrected text was being processed and displayed.
- Debugged the `perform_inference` function to ensure the corrected text was visible.
- Increased the timeout for the Ollama client to handle delays in LLM response.
- Changed the LLM model to `qwen3:0.6b`, which resolved timeout issues and improved functionality.

### Enhancements
- Integrated text-to-speech functionality to convert corrected text into `.wav` audio files.
- Updated the `perform_inference` function to generate an audio file (`corrected_output_audio.wav`) from the corrected text.

### Next Steps
- Documented the pipeline for future tasks:
  1. Obtain corrected text from the user's lips.
  2. Convert the corrected text into speech using open-source TTS to generate an audio file (preferably `.wav`).
  3. Use the FLOAT repository to generate a talking head avatar using an input image (`sam.jpg`) and the generated audio file.

## December 7, 2025

### Human-in-the-Loop Verification
- Implemented a **confidence scoring system** for the VSR model using the relative posterior probability (softmax of n-best hypotheses).
- Added an **adaptive verification policy**:
    - **High Confidence (>= 0.8)**: Automatic processing.
    - **Low Confidence (< 0.8)**: Pauses and prompts the user to **Accept**, **Edit**, or **Discard** the prediction.
- This ensures accuracy without sacrificing speed for clear inputs.

### TTS Upgrades
- Replaced the robotic `pyttsx3` engine with **Microsoft SpeechT5** (neural TTS) for high-quality, natural-sounding speech.
- Configured the system to use a **male speaker embedding ('awb')** by default.
- Implemented **model caching** to eliminate loading latency after the first inference.
- Fixed a `datasets` library incompatibility by loading the Parquet version of the speaker embeddings directly.

### Privacy & Auditing
- Designed a privacy-first architecture where all processing remains local.
- Implemented an **Audit Log** (`audit_log.json`) that records:
    - Timestamps
    - Raw VSR output
    - Confidence scores
    - User actions (processed/discarded)
- This provides transparency and a trail for future improvements.

### Refinements
- **LLM Prompt**: Optimized the Qwen system prompt to be more robust against "homophenes" (visually similar words) and to strictly follow formatting rules.
- **Codebase**: Vendored the `float_module` into the main repository for better version control and stability.

All future edits and improvements will be documented in this log.

## February 24, 2026

### Context
Project was transferred from a local disk to an external SSD. Several environment and configuration issues surfaced as a result of the move.

### Bug Fixes

#### Fix 1: `config_filename` was `null` in Hydra config
- **File**: `hydra_configs/default.yaml`
- **Issue**: `config_filename` was set to `null`, causing `cfg.config_filename` to be `None` at runtime, which triggered a `TypeError` in `os.path.isfile(None)`.
- **Fix**: Set `config_filename: configs/LRS3_V_WER19.1.ini` as the default.

#### Fix 2: Missing VSR model weights
- **Issue**: `benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth` and `benchmarks/LRS3/language_models/lm_en_subword/model.pth` were not present on the new disk.
- **Fix**: Downloaded both files from HuggingFace (`Amanvir/LRS3_V_WER19.1` and `Amanvir/lm_en_subword`) using `setup.sh` wget commands.

#### Fix 3: `ibug` face detection packages not installed
- **Issue**: `ibug-face-detection` and `ibug-face-alignment` were not present in the Python environment after the disk transfer. These packages are not on PyPI.
- **Fix**: Cloned both repos from GitHub (`hhj1897/face_detection`, `hhj1897/face_alignment`) and installed with `pip install -e .`. The `ibug/LFS` budget was exceeded for `Resnet50_Final.pth`; the file was downloaded separately and placed at `/tmp/face_detection/ibug/face_detection/retina_face/weights/Resnet50_Final.pth`.

#### Fix 4: RetinaFace detector hardcoded to `cuda:0`
- **File**: `pipelines/pipeline.py`, line 58
- **Issue**: `LandmarksDetector(device="cuda:0")` was hardcoded, ignoring the device selected at startup.
- **Fix**: Changed to `LandmarksDetector(device=str(device))` to pass through the actual runtime device.

#### Fix 5: MPS not included in device selection
- **File**: `main.py`
- **Issue**: Device selection only checked for CUDA or fell back to CPU, skipping MPS entirely.
- **Fix**: Added `elif torch.backends.mps.is_available(): device = torch.device("mps")` between the CUDA and CPU branches.

#### Fix 6: CTC prefix scorer hardcoded CUDA device detection
- **File**: `espnet/nets/ctc_prefix_score.py`
- **Issue**: `self.device` was determined by `x.is_cuda`, which returns `False` for MPS tensors, causing `self.device = cpu` while input tensors were on `mps`. This produced a `RuntimeError: Expected all tensors to be on the same device` during beam search.
- **Fix**:
  - Replaced the `is_cuda` guard with `self.device = x.device` (handles CUDA, MPS, and CPU uniformly).
  - Changed both `torch.as_tensor(xlens) - 1` occurrences to `torch.as_tensor(xlens).to(self.device) - 1` so `end_frames` lands on the correct device.
