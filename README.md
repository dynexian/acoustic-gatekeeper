# Acoustic Gatekeeper

A highly optimized 1D Convolutional Neural Network (CNN) designed to pre-filter acoustic data for distributed drone tracking arrays. This repository serves as the baseline for edge-compute nodes to identify drone rotor frequencies and drop environmental noise in single-digit milliseconds, protecting downstream Time Difference of Arrival (TDOA) solvers from matrix explosion.

## 1. The Engineering Challenge & The Nyquist Limit
Tracking drones at 250+ meters requires a massive acoustic macro-array (up to 50 meters wide). However, based on the Spatial Nyquist Limit ($\lambda/2$), wide arrays cause severe spatial aliasing on high-frequency sounds. If a raw audio feed containing wind noise, sirens, or traffic is fed directly into a GCC-PHAT/TDOA algebraic matrix, the phase shifts scatter and the tracking logic mathematically collapses. 

**The Solution:** An ultra-lightweight "Gatekeeper" neural network deployed on edge nodes (upcycled ARM64 smartphone motherboards). This model listens to 1.0-second audio bursts, instantly identifies whether a drone is present, and drops the chunk if it is just wind—ensuring the TDOA matrix only processes clean, verified targets.

## 2. Dataset Architecture & Lessons Learned
Training an acoustic model in the real world is incredibly susceptible to "Data Leakage." During development, we uncovered several traps that would artificially inflate accuracy while breaking real-world tracking:
* **The Contiguous Audio Trap:** Many datasets slice a single 5-minute recording into hundreds of 1-second chunks. Standard random splitting places adjacent chunks into both Train and Test sets, allowing the model to simply memorize the background wind speed instead of learning the drone's acoustics. We mitigated this by enforcing **Source-Interval Disjoint Splitting** and chunking by session prefix.
* **The 6-Channel Trap:** Distance-tagged field recordings often contain 6-channel microphone array data. We aggressively downmixed these to 1-channel Mono (`[16000, 1]`) to maintain tensor uniformity and prevent broadcast crashes during training.
* **Acoustic Uniformity:** The final `Master_Dataset` enforces strict physical limits: exactly 1.000s duration, 16,000 Hz sample rate, and Mono encoding. 16 kHz flawlessly captures the 100 Hz–4,000 Hz drone acoustic signature while discarding high-pitch environmental noise that wastes CPU cycles.

## 3. Model Architecture: Why 1D CNN?
A standard 2D CNN treats spectrograms like photographs, scanning for shapes in height and width. However, our MFCC arrays (`[32, 40]`) are time-series data. 

We utilized a **1D CNN** because it slides its filter strictly *across time*. It observes all 40 frequency bands simultaneously, scanning left-to-right to find the persistent, continuous mechanical whine of a drone rotor, contrasting it against the chaotic, temporary spikes of wind noise. 
* **Aggressive Dropout:** A `Dropout(0.5)` layer was mandatory to prevent the network from memorizing ambient noise, forcing genuine generalization.
* **Parameters:** 34,785 (Highly optimized for edge hardware).

## 4. Performance & Edge Optimization
The model was compiled and optimized into TensorFlow Lite (`.tflite`) using Dynamic Range Quantization, crushing the 32-bit floating-point weights down to 8-bit integers.
* **Model Size:** 43.42 KB (Small enough to load entirely into L1/L2 CPU cache).Untitled
* **Inference Speed:** 0.03 milliseconds per frame on standard CPU.
* **Precision:** 99.0% True Positive drone identification with a near-zero False Positive rate against ESC-50 environmental noise.

## 5. Supporting Documentation
Detailed mathematical models regarding the spatial aliasing constraints and the TDOA Apache Spark pipeline are located in the `docs/` folder (see `acoustic_radar_methodology.pdf`).

## 6. Project Scope & Dataset Attributions
This repository is strictly for **educational, experimental, and learning purposes**. It is not currently intended for commercial deployment. The machine learning models herein were trained using data from the following publicly available acoustic datasets:
* **ITU ARIS Lab. Acoustic Drone Dataset:** Muhacıroğlu, İ. M. and Akgül, T. (2026). Published under CC-BY-4.0. Contains continuous outdoor drone flight intervals and background environmental noise.
* **Saraleemadi Drone Audio Dataset:** Binary and multiclass (Parrot Bebop / Mambo) acoustic flight signatures.
* **Gautam Acoustic Drone Dataset (Kaggle):** Multi-distance drone field recordings and ambient environmental audio.
