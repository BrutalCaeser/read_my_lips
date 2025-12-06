# Chaplin

![Chaplin Thumbnail](./thumbnail.png)

A visual speech recognition (VSR) tool that reads your lips in real-time and types whatever you silently mouth. Runs fully locally.

Relies on a [model](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages?tab=readme-ov-file#autoavsr-models) trained on the [Lip Reading Sentences 3](https://mmai.io/datasets/lip_reading/) dataset as part of the [Auto-AVSR](https://github.com/mpc001/auto_avsr) project.

Watch a demo of Chaplin [here](https://youtu.be/qlHi0As2alQ).

## Setup

1. Clone the repository, and `cd` into it:
   ```sh
   git clone https://github.com/amanvirparhar/chaplin
   cd chaplin
   ```
2. Run the setup script...
   ```sh
   ./setup.sh
   ```
   ...which will automatically download the required model files from Hugging Face Hub and place them in the appropriate directories:
   ```
   chaplin/
   ├── benchmarks/
       ├── LRS3/
           ├── language_models/
               ├── lm_en_subword/
           ├── models/
               ├── LRS3_V_WER19.1/
   ├── ...
   ```
3. Install and run `ollama`, and pull the [`qwen3:4b`](https://ollama.com/library/qwen3:4b) model.
4. Install [`uv`](https://github.com/astral-sh/uv).

# Chaplin: The Silent-to-Voice Avatar Interface

![Chaplin Thumbnail](./thumbnail.png)

**Chaplin** is an end-to-end assistive communication tool designed to empower mute individuals or those who prefer silent communication. It reads your lips in real-time, understands what you are saying, and projects it through a lifelike talking avatar with a synthesized voice.

The entire pipeline runs **fully locally** on your machine, ensuring privacy and low latency.

---

## 🚀 The 3-Stage Pipeline

Chaplin transforms silent lip movements into a full audio-visual experience through three distinct stages:

```mermaid
graph LR
    A[User Mouths Words] -->|Video Feed| B(Stage 1: VSR);
    B -->|Raw Text| C(Stage 2: Correction & TTS);
    C -->|Audio & Text| D(Stage 3: Talking Head);
    D -->|Final Video| E[Avatar Speaks];
```

### 1. Visual Speech Recognition (VSR)
*   **What it does**: Captures video from your webcam and uses a deep learning model to "read" your lips.
*   **Technology**: Based on [Auto-AVSR](https://github.com/mpc001/auto_avsr) and trained on the LRS3 dataset.
*   **Output**: A raw, often imperfect, text transcription (e.g., "HELLO WORLD HOW ARE U").

### 2. LLM Correction & Speech Synthesis
*   **What it does**: 
    *   **Correction**: Passes the raw text to a local Large Language Model (LLM) to fix typos, grammar, and context (e.g., "Hello world, how are you?").
    *   **Synthesis**: Converts the corrected text into high-quality speech using a neural Text-to-Speech (TTS) engine.
*   **Technology**: 
    *   **LLM**: [Ollama](https://ollama.com/) running `qwen3`.
    *   **TTS**: Microsoft SpeechT5 (via Hugging Face Transformers).
*   **Output**: A polished sentence and a corresponding `.wav` audio file.

### 3. Talking Head Generation
*   **What it does**: Animates a static reference image (an avatar) to lip-sync perfectly with the generated audio.
*   **Technology**: [FLOAT](https://github.com/deepbrainai-research/float) (Flow Matching for Audio-driven Talking Portrait).
*   **Output**: A video file where the avatar speaks your words with the correct emotion and lip movements.

---

## 🛠️ Setup

### Prerequisites
*   **OS**: macOS (Apple Silicon recommended for performance) or Linux.
*   **Python**: 3.10 or higher.
*   **Tools**: `git`, `ffmpeg` (for video processing).

### Installation

1.  **Clone the repository**:
    ```sh
    git clone https://github.com/amanvirparhar/chaplin
    cd chaplin
    ```

2.  **Run the setup script**:
    This downloads the VSR models and organizes the directory structure.
    ```sh
    ./setup.sh
    ```

3.  **Install & Configure Ollama**:
    *   Download [Ollama](https://ollama.com/).
    *   Pull the Qwen model:
        ```sh
        ollama pull qwen3:4b
        ```

4.  **Install `uv` (Python Package Manager)**:
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

5.  **Install Python Dependencies**:
    The project uses `uv` to manage dependencies automatically. Ensure you have the requirements file ready (included in the repo).

---

## 🎮 Usage

1.  **Start the Application**:
    Run the main script using `uv`. This will set up the environment and launch the webcam feed.
    ```sh
    uv run --with-requirements requirements.txt --python 3.12 main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
    ```

2.  **Record Your Speech**:
    *   Focus on the webcam window.
    *   Press and hold the **`Option`** key (macOS) or **`Alt`** key (Windows/Linux) to toggle recording.
    *   Mouth your sentence silently.
    *   Press the key again to stop recording.

3.  **Watch the Magic**:
    *   **Terminal**: You'll see the raw VSR output and the corrected text.
    *   **Audio**: The system generates the speech audio.
    *   **Video**: A window will pop up showing your avatar speaking the sentence!

4.  **Exit**:
    Press **`q`** in the webcam window to quit the application.

---

## 🧩 Customization

*   **Change the Avatar**: Replace `float_module/assets/sam_altman.webp` with any portrait image you like to change the talking head identity.
*   **Change the Voice**: The TTS engine uses speaker embeddings. You can modify `pipelines/pipeline.py` to select different voices from the CMU Arctic dataset.

---

## 📄 License
This project combines multiple open-source technologies. Please refer to the individual licenses of [Auto-AVSR](https://github.com/mpc001/auto_avsr), [FLOAT](https://github.com/deepbrainai-research/float), and the datasets used.
