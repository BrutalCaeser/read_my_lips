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

All future edits and improvements will be documented in this log.
