## **Read My Lips**

Team members:

* Yashvardhan Gupta \- 002853544  
* Sai Krishna Reddy Maligireddy \-  002593642

## **1\. Project Proposal**

### **1.1 Introduction**

**Goal.** Build a real-time, privacy-preserving assistive system that converts *silent lip movements* of a non-speaking person into intelligible spoken output and a synchronized talking avatar. The system will (a) infer text from silent video of a user’s mouth, (b) synthesize natural speech from the text, and (c) render a temporally-synchronized, realistic talking face so the listener experiences natural, multimodal communication.

**What we are working on.** The prototype integrates four components into a low-latency pipeline: face & mouth capture and normalization; a visual-speech recognition (VSR; “lipreading”) model producing text; a lightweight language post-processor to improve readability; an open-source TTS model to synthesize natural speech; and an audio-driven talking-head module (lip-sync / facial animation) to generate a synchronized avatar. 

**Why are we working on this?** Current augmentative and alternative communication tools (AAC boards, text entry) are slow and interrupt conversational flow. A low-latency silent-speech avatar can restore conversational immediacy for people with speech loss (e.g., ALS, laryngectomy, severe aphasia), increasing autonomy and social participation. The project also advances robust AV modelling in constrained realistic settings and demonstrates responsible deployment practices (local processing, visible watermarking, human-in-loop verification).

**Fields impacted.** Assistive technology and accessibility; multimodal machine learning (computer vision \+ speech); human-computer interaction (HCI); real-time systems engineering; and ethical AI / privacy engineering.

---

### **1.2 Previous Work and Contributions**

We discuss three representative works that directly inform our approach.

**(2)**[https://github.com/mpc001/auto\_avsr](https://github.com/mpc001/auto_avsr) We use this model for lip reading which is built upon

 **AV-HuBERT / Self-supervised audio-visual speech features (e.g., AV-HuBERT et al., 2021–2022).**  
*Contribution.* AV-HuBERT and related self-supervised audio-visual methods learn powerful multimodal representations from large unlabeled video corpora; when fine-tuned they significantly improve VSR quality and robustness.  
*What we liked.* Self-supervised pretraining reduces labelled data needs and improves generalization to noisy and low-resource scenarios; the learned features provide a strong backbone for downstream decoders.  
*Limitations.* Pretrained models are large; efficient real-time deployment requires compression/quantization or smaller student models. Direct lipreading accuracy still degrades with rapid head motion and low resolution.

**(3) [https://github.com/deepbrainai-research/float](https://github.com/deepbrainai-research/float)** \- We use this repo for Talking head video generation. 

---

#### **Our planned contributions**

We will produce a complete, demonstrable prototype and the following technical contributions:

1. **Robust VSR pipeline for realistic short utterances.**  
2. **Low-latency TTS integration with streaming audio synthesis.**  
   * *Methodology:* Use an open-source streaming TTS (VITS or a smaller student model) hosted locally; implement short-utterance streaming to minimize delay. Provide selectable voices and prosody conditioning.  
   * *Motivation & alternatives:* Precomputed audio is not feasible for live interaction. Server-side hosted TTS reduces local compute needs but raises privacy concerns; we will prefer local inference where GPU is available and provide a clear privacy option.  
3. **Audio→avatar rendering (lip-sync \+ head motion).**  
   * *Methodology:* Use FLOAT  
4. **Human-in-loop verification and privacy design.**  
   * *Methodology:* Present predicted text (subtitle) and a confidence bar; allow caregiver or user to accept or edit before final speech output when confidence \< threshold. All processing by default runs locally; exports include an audit JSON recording timestamps/confidences.  
   * *Motivation & alternatives:* Automatic broadcasting without verification risks errors. An always-manual confirm flow is slower; we choose an adaptive policy: automatic speak when confidence is high, require review otherwise.

For each contribution we will justify parameter choices (window sizes, thresholds) via ablation during development.

---

### **1.3 Expected Results / Preliminary Results**

**Problem instances and expectations.** We consider three deployment regimes (increasing difficulty) and give expected performance ranges informed by literature:

1. **Controlled lab conditions (frontal camera, good lighting, neutral background, short prearranged phrases).**  
   * *Expectation:* Word error rate (WER) ≈ 20–35% on 1–3s utterances after fine-tuning and LM rescoring; end-to-end latency ≲1.5s; listener comprehension rate (human transcription of TTS output) \> 85%. These numbers reflect constrained lab benchmarks and will enable convincing demos.  
2. **Semi-controlled real use (home lighting, moderate head motion, natural phrases).**  
   * *Expectation:* WER ≈ 35–60%; latency ≲2.5s with optimizations; comprehension rate 70–85% depending on utterance length and phrase complexity. Errors will concentrate on low-contrast phonemes and out-of-vocabulary terms.  
3. **In-the-wild (poor lighting, occlusion, rapid head turns).**  
   * *Expectation:* WER \> 60% without per-user adaptation. We expect reduced accuracy but the system will still produce subtitle suggestions and a confidence flag; in low-confidence situations the UI will insist on human confirmation.

**Strategies to improve performance.**

* *Personalized adaptation:* fine-tune the final decoder on a small private dataset of the intended user (5–15 minutes) to reduce domain gap — anticipated WER reduction of 10–25 percentage points.  
* *Multimodal fusion:* when available, incorporate residual audio (e.g., breath sounds) or head pose signals for disambiguation.  
* *Confidence calibration and verification:* use calibrated confidence scores to gate speech output and reduce harmful miscommunications.  
* *Model compression:* convert TTS and VSR decoders to ONNX/quantized formats for speed; use half-precision inference where safe.

**Evaluation plan and metrics.** Evaluation will report WER (VSR), listener comprehension (human transcription accuracy of produced speech), MOS for TTS naturalness, lip-sync quality (LSE-C / human rating), and end-to-end latency. We will test on public datasets (LRS3 / GRID / LRW) for baseline metrics and on a small curated private test set representing the target user domain.

