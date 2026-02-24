# Chaplin Project: Visual Speech Recognition & Talking Head Generation Pipeline

## 1. Project Overview
**Chaplin** is a local, real-time Visual Speech Recognition (VSR) tool that has been extended to include a talking head generation capability. The complete pipeline allows a user to:
1.  **Speak silently** into a webcam (Lip Reading).
2.  **Transcribe** the visual speech into text using a VSR model.
3.  **Correct** the raw transcription using a local Large Language Model (LLM).
4.  **Synthesize** the corrected text into speech (TTS).
5.  **Generate** a talking head video synchronized with the synthesized audio.

---

## 2. Pipeline Architecture

### Stage 1: Visual Speech Recognition (VSR)
*   **Input**: Real-time video feed from the user's webcam.
*   **Process**:
    *   The user toggles recording with the `Alt`/`Option` key.
    *   Video frames are captured and passed to the `InferencePipeline` (wrapping an `AVSR` model).
    *   The model (based on Auto-AVSR/E2E Transformer) predicts raw text from the lip movements.
    *   **Confidence Scoring**: The model calculates a relative posterior probability score (0.0-1.0) based on the softmax of the top beam search hypotheses.
*   **Output**: 
    *   Raw text transcription (e.g., "HELLO WORLD").
    *   Confidence score (e.g., 0.85).

### Stage 1.5: Human-in-the-Loop Verification
*   **Input**: Raw text and Confidence score from Stage 1.
*   **Process**:
    *   The system checks the confidence against a threshold (default `0.8`).
    *   **High Confidence**: Automatically proceeds to Stage 2.
    *   **Low Confidence**: Pauses and prompts the user via the terminal to:
        *   **(a)ccept**: Use the text as is.
        *   **(e)dit**: Manually correct the text.
        *   **(d)iscard**: Ignore the input.
*   **Output**: Verified text ready for correction.

### Stage 2: Text Correction & Speech Synthesis
*   **Input**: Verified text from Stage 1.5.
*   **Process**:
    *   **Correction**: The text is sent to a local Ollama instance running the `qwen3` model. A refined system prompt instructs the LLM to fix mistranscriptions (handling homophenes) and add punctuation without altering the meaning.
    *   **TTS**: The corrected text is converted to speech using **Microsoft SpeechT5**.
        *   Uses a specific male speaker embedding (`cmu_us_awb_arctic`).
        *   Models are cached in memory for low-latency generation.
*   **Output**:
    *   Corrected text (e.g., "Hello, world.").
    *   High-quality audio file (`corrected_output_audio.wav`).

### Stage 3: Talking Head Generation (FLOAT Integration)
*   **Input**:
    *   A reference image (e.g., `sam_altman.webp`).
    *   The audio file from Stage 2.
*   **Process**:
    *   The system initializes the **FLOAT** (Flow Matching for Audio-driven Talking Portrait) model.
    *   It uses `Wav2Vec2` for audio feature extraction and emotion recognition.
    *   A flow-matching transformer generates motion latents based on the audio.
    *   A style-based decoder renders the final video frames, warping the reference image to match the predicted motion.
*   **Output**: A video file (`output_video_{sequence}.mp4`) of the reference image speaking the corrected text.

---

## 3. Privacy & Auditing

To ensure safety and transparency, Chaplin includes a robust auditing system:

*   **Local Processing**: All VSR, LLM, and TTS inference happens locally on the device. No audio or video data is sent to the cloud.
*   **Audit Log**: A JSON file (`audit_log.json`) records every interaction, including:
    *   Timestamp
    *   Raw VSR output
    *   Confidence score
    *   Threshold used
    *   Action taken (processed/discarded)

---

## 4. Integration Challenges & Solutions

Integrating the `float` repository into the existing `chaplin` project on a macOS (M4 chip) environment presented several challenges. Below is a log of the errors encountered and their resolutions.

### Error 1: Missing Dependencies
**Issue**: The `float` module required several libraries not present in the original `chaplin` environment.
**Error**: `ModuleNotFoundError: No module named 'librosa'` (and others).
**Solution**:
*   Updated `requirements.txt` to include: `librosa`, `transformers`, `albumentations`, `albucore`, `torchdiffeq`, `timm`, `face_alignment`, `flow-vis`, `pandas`, `tqdm`, `matplotlib`, `pyyaml`.
*   Ran `uv` to install these dependencies into the virtual environment.

### Error 2: Argument Parsing Conflict
**Issue**: The `float` module's `InferenceOptions` class used `argparse` to parse command-line arguments. This conflicted with `hydra`, which `chaplin` uses for its own configuration.
**Error**: `main.py: error: unrecognized arguments: config_filename=...`
**Solution**:
*   Modified `float_module/options/base_options.py` to accept an optional `args` list in the `parse` method.
*   Updated `chaplin.py` to call `InferenceOptions().parse(args=[])`, forcing it to ignore the global command-line arguments.

### Error 3: CUDA/Device Mismatch on macOS
**Issue**: The `float` code was hardcoded to expect CUDA or didn't handle the MPS (Metal Performance Shaders) device correctly for `face_alignment`.
**Error**: `AssertionError: Torch not compiled with CUDA enabled` (in `face_alignment`).
**Solution**:
*   Modified `float_module/generate.py` to explicitly pass `device='mps'` (or `'cpu'`) to the `face_alignment.FaceAlignment` constructor based on availability.

### Error 4: Missing MPS Operator (QR Decomposition)
**Issue**: The `torch.linalg.qr` operator is not yet implemented for the MPS backend in PyTorch.
**Error**: `NotImplementedError: The operator 'aten::linalg_qr.out' is not currently implemented for the MPS device.`
**Solution**:
*   **Fallback**: Added `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` in `chaplin.py` to allow PyTorch to fall back to CPU for missing operators.
*   **Explicit Cast**: Modified `float_module/models/float/styledecoder.py` to explicitly move the tensor to CPU before calling `qr`, then move the result back to the original device.

### Error 5: Attention Implementation Conflict
**Issue**: Newer `transformers` versions default to `sdpa` (Scaled Dot Product Attention), which does not support `output_attentions=True`, a requirement for the `Wav2VecModel` usage in this project.
**Error**: `ValueError: The output_attentions attribute is not supported when using the attn_implementation set to sdpa.`
**Solution**:
*   Modified `float_module/models/float/FLOAT.py` to load the `Wav2VecModel` with `attn_implementation="eager"`.

### Error 6: Hardcoded CUDA Calls
**Issue**: The `styledecoder.py` file contained a hardcoded `.cuda()` call for creating a grid tensor.
**Error**: `AssertionError: Torch not compiled with CUDA enabled`.
**Solution**:
*   Replaced `.cuda()` with `.to(input.device)` in `float_module/models/float/styledecoder.py` to ensure compatibility with both MPS and CPU.

### Error 7: SpeechT5 Dataset Security Restriction
**Issue**: The `datasets` library blocked the execution of the custom loading script for `Matthijs/cmu-arctic-xvectors` due to security policies (`trust_remote_code=True` is deprecated/unsafe).
**Error**: `ValueError: trust_remote_code is not supported anymore.`
**Solution**:
*   Modified `pipelines/pipeline.py` to load the pre-converted **Parquet** version of the dataset directly from the Hugging Face Hub, bypassing the need for a local execution script.

---

### Error 8: `config_filename` null in Hydra config (post-SSD transfer)
**Issue**: After transferring the project to an external SSD, `hydra_configs/default.yaml` had `config_filename: null`, so `cfg.config_filename` was `None` at runtime.
**Error**: `TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType`
**Solution**:
*   Set `config_filename: configs/LRS3_V_WER19.1.ini` as the default in `hydra_configs/default.yaml`.

### Error 9: Missing model weight files (post-SSD transfer)
**Issue**: The `.pth` weight files for the VSR model and language model were absent on the new disk. Only the `.json` config files had been committed to the repository.
**Error**: `FileNotFoundError: benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth`
**Solution**:
*   Re-downloaded both files via the existing `setup.sh` script from HuggingFace (`Amanvir/LRS3_V_WER19.1` — 955 MB, `Amanvir/lm_en_subword` — 205 MB).

### Error 10: `ibug` packages missing (post-SSD transfer)
**Issue**: `ibug-face-detection` and `ibug-face-alignment` were installed as editable packages pointing to a path that no longer existed after the disk transfer. They are not available on PyPI.
**Error**: `ModuleNotFoundError: No module named 'ibug'`
**Solution**:
*   Cloned `github.com/hhj1897/face_detection` and `github.com/hhj1897/face_alignment`.
*   Installed both with `pip install -e .`.
*   The ibug LFS budget was exceeded for `Resnet50_Final.pth` (110 MB); the file was obtained separately and placed at `/tmp/face_detection/ibug/face_detection/retina_face/weights/Resnet50_Final.pth`.

### Error 11: RetinaFace detector hardcoded to `cuda:0`
**Issue**: `pipelines/pipeline.py` passed `device="cuda:0"` directly to `LandmarksDetector`, ignoring the device resolved at startup.
**Error**: `AssertionError: Torch not compiled with CUDA enabled`
**Solution**:
*   Changed line 58 of `pipelines/pipeline.py` from `LandmarksDetector(device="cuda:0")` to `LandmarksDetector(device=str(device))`.
*   Updated `main.py` to include MPS in the device selection chain: CUDA → MPS → CPU.

### Error 12: CTC prefix scorer device mismatch on MPS
**Issue**: `espnet/nets/ctc_prefix_score.py` determined `self.device` using `x.is_cuda`, which returns `False` for MPS tensors, silently setting `self.device = cpu`. All internally created tensors (`idx_b`, `idx_bo`, `end_frames`, etc.) were placed on CPU while inputs were on `mps`.
**Error**: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!`
**Solution**:
*   Replaced the `is_cuda` conditional block with `self.device = x.device` — a single line that handles CUDA, MPS, and CPU uniformly.
*   Fixed `self.end_frames = torch.as_tensor(xlens).to(self.device) - 1` (two occurrences) so the index tensor is created on the correct device.

---

## 5. Conclusion
The project now successfully runs a complete end-to-end pipeline on macOS (Apple Silicon). It leverages the power of local LLMs for text correction and state-of-the-art generative models for video synthesis, all within a unified Python environment.
