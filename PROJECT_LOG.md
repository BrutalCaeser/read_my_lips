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
