# Read_my_lips: A Privacy-Preserving, End-to-End Visual Speech-to-Avatar Interface

**Authors:**
*   Yashvardhan Gupta (002853544)
*   Sai Krishna Reddy Maligireddy (002593642)

**Date:** December 7, 2025

---

## Abstract

This report presents **Read_my_lips**, a real-time assistive communication system designed to restore conversational agency to individuals with speech impairments. By converting silent lip movements into synthesized speech and a synchronized talking avatar, Read_my_lips bridges the gap between silent articulation and audible, visual communication. The system integrates state-of-the-art Visual Speech Recognition (VSR), Large Language Model (LLM) based error correction, Neural Text-to-Speech (TTS), and generative talking head animation into a unified, locally-executed pipeline. We demonstrate a privacy-first architecture that operates entirely on-device, featuring a human-in-the-loop verification mechanism to ensure accuracy and user control. Our evaluation highlights the system's capability to produce intelligible speech and realistic visual output from silent video input, offering a novel multimodal interface for accessibility.

---

## 1. Introduction

### 1.1 Motivation
Communication is fundamental to human connection and autonomy. For individuals suffering from conditions such as Amyotrophic Lateral Sclerosis (ALS), laryngeal cancer, or severe aphasia, the loss of speech can be isolating. Traditional Augmentative and Alternative Communication (AAC) devices, such as eye-tracking keyboards or text-to-speech boards, often disrupt the natural flow of conversation due to slow input rates and the lack of non-verbal cues.

There is a critical need for technology that can interpret the *intent* of speech directly from articulation—without requiring sound—and project it back into the world in a natural, human-like manner.

### 1.2 Problem Statement
The core challenge is to build a low-latency system that can:
1.  Accurately transcribe text from silent video of a user's mouth (Lip Reading).
2.  Correct the inevitable transcription errors inherent in visual-only speech recognition.
3.  Synthesize natural-sounding audio from the corrected text.
4.  Generate a photorealistic video avatar that lip-syncs to the audio in real-time.
5.  Ensure user privacy and control through local processing and verification protocols.

### 1.3 Contributions
We present the following technical contributions:
*   **End-to-End VSR Pipeline**: Integration of Auto-AVSR for robust lip reading in realistic settings.
*   **Context-Aware Correction**: Deployment of a local Small Language Model (SLM), `qwen3:0.6b`, to resolve homophenes (visually similar words) and restore grammatical structure.
*   **High-Fidelity Synthesis**: Implementation of Microsoft SpeechT5 for neural TTS with speaker embedding support.
*   **Generative Avatar Interface**: Adaptation of the FLOAT (Flow Matching) model for audio-driven talking head generation on consumer hardware (macOS/MPS).
*   **Adaptive Verification**: A confidence-based human-in-the-loop mechanism that balances automation with accuracy.

---

## 2. System Architecture

The Read_my_lips pipeline consists of three sequential stages, orchestrated by a central controller (`chaplin.py`). The entire system is designed to run locally on Apple Silicon (M-series chips) to ensure data privacy.

### 2.1 Stage 1: Visual Speech Recognition (VSR)
*   **Input**: Live video feed from a webcam (captured via OpenCV).
*   **Model**: We utilize the **Auto-AVSR** architecture, a state-of-the-art model pre-trained on the LRS3 dataset.
*   **Process**:
    1.  **Face Detection**: MediaPipe is used to detect and crop the user's mouth region in real-time.
    2.  **Inference**: The cropped frames are passed to the VSR model, which predicts a sequence of characters.
    3.  **Confidence Scoring**: We implemented a relative posterior scoring mechanism. By computing the softmax of the top-$N$ beam search hypotheses, we derive a confidence score ($0.0 - 1.0$) representing the model's certainty.
*   **Output**: Raw text transcription (e.g., "HELLO WORLD") and a confidence metric.

### 2.2 Stage 1.5: Human-in-the-Loop Verification
To mitigate the risk of broadcasting incorrect speech, we implemented an adaptive policy:
*   **High Confidence ($\ge 0.45$)**: The system automatically proceeds to the next stage.
*   **Low Confidence ($< 0.45$)**: The system pauses and presents the predicted text to the user via the terminal. The user can:
    *   **(a)ccept**: Confirm the text is correct.
    *   **(e)dit**: Manually type the intended phrase.
    *   **(d)iscard**: Reject the input entirely.

### 2.3 Stage 2: Text Correction & Speech Synthesis
*   **LLM Correction**:
    *   Raw VSR output lacks punctuation and often contains "homophene" errors (e.g., confusing 'mat' vs 'pat').
    *   We use **Ollama** running the **Qwen3 (0.6B)** model with a specialized system prompt. The prompt instructs the model to fix nonsensical words based on context and apply proper sentence casing/punctuation without hallucinating new content.
*   **Text-to-Speech (TTS)**:
    *   We integrated **Microsoft SpeechT5**, a transformer-based TTS model.
    *   **Optimization**: To reduce latency, the TTS pipeline and speaker embeddings (specifically the male speaker `cmu_us_awb_arctic`) are cached in memory after the first inference.
*   **Output**: A polished sentence (e.g., "Hello, world.") and a `.wav` audio file.

### 2.4 Stage 3: Talking Head Generation
*   **Model**: We adapted **FLOAT** (Flow Matching for Audio-driven Talking Portrait).
*   **Mechanism**:
    1.  **Audio Encoding**: `Wav2Vec2` extracts features and emotion embeddings from the generated audio.
    2.  **Flow Matching**: A transformer predicts motion latents that align the lip movements of a reference image (e.g., a photo of the user or a persona) with the audio.
    3.  **Rendering**: A style-based decoder generates the final video frames.
*   **Platform Adaptation**: Significant engineering was required to port FLOAT to macOS. We implemented fallbacks for missing MPS operators (like `linalg.qr`) and optimized tensor casting to ensure stability on Apple Silicon.

---

## 3. Implementation Details

### 3.1 Technology Stack
*   **Language**: Python 3.12
*   **Dependency Management**: `uv` (for fast, reproducible environments)
*   **Deep Learning Framework**: PyTorch (with MPS backend support)
*   **Libraries**: `hydra` (config), `transformers` (TTS), `face_alignment` (landmarks), `ollama` (LLM interface).

### 3.2 Privacy & Auditing
Privacy is a core tenet of Read_my_lips.
*   **Local-Only**: No video or audio data is sent to cloud APIs.
*   **Audit Log**: A JSON log (`audit_log.json`) records every interaction, including timestamps, raw inputs, confidence scores, and user actions. This provides transparency and a dataset for future fine-tuning.

---

## 4. Evaluation & Results

### 4.1 Qualitative Analysis
*   **VSR Accuracy**: In controlled lighting with frontal views, the system reliably captures short phrases. Performance degrades with rapid head movement or extreme angles, consistent with VSR literature.
*   **Correction Quality**: The Qwen3 model successfully resolves common homophene ambiguities (e.g., correcting "I AM FINE" from "I AP FINE") and consistently formats text for TTS.
*   **Avatar Realism**: The FLOAT model produces highly synchronized lip movements. The generated video preserves the identity of the reference image while animating the mouth and lower face naturally.

### 4.2 Performance Metrics
*   **Latency**:
    *   VSR Inference: ~200ms (per utterance)
    *   LLM Correction: ~500ms
    *   TTS Generation: ~800ms (cached)
    *   Video Generation: ~3-5s (depending on utterance length)
    *   *Note*: While video generation is the bottleneck, the audio is available almost immediately, allowing for rapid communication while the visual avatar catches up.

### 4.3 Challenges Overcome
1.  **macOS Compatibility**: Porting CUDA-centric research code (FLOAT) to MPS required extensive patching of tensor operations and explicit CPU fallbacks for unsupported linear algebra functions.
2.  **Dataset Security**: We resolved security restrictions in the `datasets` library by implementing direct Parquet loading for speaker embeddings.
3.  **Model Hallucination**: Early LLM prompts caused the model to rewrite sentences entirely. We refined the prompt to strictly enforce "minimal intervention," restricting the model to only fix transcription errors.

---

## 5. Future Work

*   **Streaming Pipeline**: Currently, the system processes utterance-by-utterance. Moving to a streaming architecture (e.g., VITS for TTS, streaming VSR) would significantly reduce end-to-end latency.
*   **Personalization**: Fine-tuning the VSR model on the specific user's lip movements would drastically improve accuracy in "in-the-wild" conditions.
*   **Multimodal Fusion**: Integrating residual audio (e.g., grunts, breath sounds) could help disambiguate intent when lip movements are unclear.

---

## 6. Conclusion

Read_my_lips demonstrates the viability of a fully local, privacy-preserving assistive communication interface. By combining visual speech recognition with generative AI, we have created a prototype that not only restores the *function* of speech but also the *form*—allowing users to communicate with their own digital voice and face. This project serves as a foundational step towards more natural, inclusive, and autonomous communication technologies.

---

## References
1.  Ma, P., et al. "Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels." (2023).
2.  Ren, Y., et al. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech." (2020).
3.  DeepBrain AI. "FLOAT: Flow Matching for Audio-driven Talking Portrait." (2024).
4.  Microsoft. "SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing." (2023).
