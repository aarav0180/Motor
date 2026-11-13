# 🧠 Vibration Anomaly Detection: Model Architecture & Engineering Plan

> **Objective**: Train a high-precision, speed-invariant, one-class anomaly detector on healthy vibration data using Google Colab, optimize it via INT8 quantization, and deploy it to a Raspberry Pi for real-time edge motor condition monitoring.

---

## 1. Problem Formulation & One-Class Philosophy

In industrial predictive maintenance, supervised binary classification (training on 50% healthy and 50% broken motors) is impractical:
* **Asymmetric Data**: Motors run in a healthy state 99.9% of their lifespan. Inducing destructive failures across hundreds of machines is cost-prohibitive and dangerous.
* **Open-World Faults**: Real-world defects take countless unpredictable forms (chipped tooth, cage crack, flaking, thermal expansion, mounting looseness, eccentric rotor, blade mass loss). A supervised classifier only catches defects it has explicitly seen.

### The Solution: One-Class Deep Reconstruction
The model is trained **strictly on healthy motor operations** ($X_{\text{healthy}}$). It learns a compact latent representation of the mechanical baseline.

$$\hat{X} = f_{\text{decoder}}(f_{\text{encoder}}(X))$$

* When input $X$ is **Healthy**: The autoencoder reconstructs it with high fidelity $\rightarrow \text{Error} \approx 0$.
* When input $X$ contains **Defects**: Unseen shock harmonics, impact spikes, or sidebands cannot be reconstructed through the bottleneck $\rightarrow \text{Error} \gg 0 \implies \mathbf{ALERT}$.

---

## 2. Why a 1D-CNN Autoencoder?

| Model Family | Strengths | Critical Weaknesses for Vibration | Verdict |
|---|---|---|---|
| **Simple RMS Threshold** | Ultra-lightweight ($O(1)$) | Blind to spectral distribution; speed changes cause false alarms | Baseline only |
| **Isolation Forest** | Fast; good on scalar feature tables | Cannot model continuous spectral shapes or localized harmonic sidebands | Classical fallback |
| **Fully-Connected MLP** | Learns global interactions | Ignores spatial/spectral locality; parameters scale quadratically ($O(N^2)$) | Suboptimal |
| **1D-CNN Autoencoder** | **Captures localized harmonic peaks, sideband spacing, and impact bursts; parameter-efficient; fast on ARM CPU** | Requires clean 1-D convolution architecture | **Selected Architecture** |

### Physical Justification
Vibration defects in bearings and rotating machinery manifest as:
1. **Periodic shock pulses**: When a rolling element strikes a spall on the inner race (BPFI) or outer race (BPFO), generating high-frequency resonant ring-downs.
2. **Sidebands**: Modulations around harmonic peaks ($\pm f_{\text{run}}$).

1D Convolutional kernels act as **adaptive digital filters**. They sweep across spectral bins to identify narrow harmonic peaks and modulation sidebands irrespective of their exact global amplitude.

---

## 3. Network Architecture Specifications

```
Input: 512-point Welch PSD / Normalized Spectrum (0 to 500 Hz or Order-Normalized)
  │
  ▼
[Conv1D Layer 1] ──► 1 channel → 32 filters, Kernel=7, Stride=2, Padding=3
  │                  BatchNorm1d + LeakyReLU(0.2) + MaxPool1d(2)  → (32, 128)
  ▼
[Conv1D Layer 2] ──► 32 filters → 64 filters, Kernel=5, Stride=2, Padding=2
  │                  BatchNorm1d + LeakyReLU(0.2) + MaxPool1d(2)  → (64, 32)
  ▼
[Conv1D Layer 3] ──► 64 filters → 128 filters, Kernel=3, Stride=2, Padding=1
  │                  BatchNorm1d + LeakyReLU(0.2)                 → (128, 16)
  ▼
[Bottleneck]     ──► Flatten (2048) → Linear(2048, 16) → Linear(16, 2048)
  │                  Latent Space Dimension = 16 (Extreme Compression)
  ▼
[ConvTrans1D 1]  ──► 128 filters → 64 filters, Kernel=3, Stride=2, Padding=1
  │                  BatchNorm1d + LeakyReLU(0.2)                 → (64, 32)
  ▼
[ConvTrans1D 2]  ──► 64 filters → 32 filters, Kernel=5, Stride=2, Padding=2
  │                  BatchNorm1d + LeakyReLU(0.2)                 → (32, 128)
  ▼
[ConvTrans1D 3]  ──► 32 filters → 1 channel, Kernel=7, Stride=4, Output_padding=3
  │                  Sigmoid()                                    → (1, 512)
  ▼
Output: 512-point Reconstructed Spectrum [0, 1]
```

### Parameter Budget & Computational Footprint
* **Total Parameters**: $\approx 78,000$ (Compact).
* **Model Size (FP32)**: $\approx 312\text{ KB}$.
* **Model Size (INT8 Quantized)**: $\approx 85\text{ KB}$.
* **Inference Latency on Raspberry Pi 4 (Cortex-A72)**: $\approx 3.2\text{ ms}$ per 10-second window.

---

## 4. Speed-Invariance & Order-Tracking Strategy

### The Core Challenge
When an operator switches the motor speed from **Low (25 Hz)** to **High (55 Hz)**, standard models raise false alarms because:
1. Centrifugal force increases with the square of speed ($F \propto \omega^2$), raising overall vibration energy.
2. Harmonic peaks shift rightward in frequency space ($25\text{ Hz} \rightarrow 55\text{ Hz}$).

### The Engineering Solution: Order Normalization
Instead of computing standard frequency in Hertz ($f$), we normalize the frequency axis by the fundamental rotational speed ($f_{\text{run}}$):

$$\text{Order} = \frac{f}{f_{\text{run}}}$$

* **Order 1.0**: Represents the fundamental motor speed ($1\times$), regardless of whether the motor is spinning at 1500 RPM or 3000 RPM.
* **Order 2.0**: Represents the second harmonic ($2\times$ unbalance/misalignment).
* **Order 3.0**: Represents the third harmonic ($3\times$).

By resampling the spectrum onto a uniform order grid $[0, 10.0]$ with 512 bins, **the spectrum looks identical at every healthy speed**. A speed shift no longer triggers an anomaly alarm!

---

## 5. Healthy Data Augmentation Strategy

To prevent the autoencoder from overfitting to a single steady-state recording in Colab, we inject physics-informed synthetic variations into healthy training windows:

1. **Spectral Jitter ($\pm 3\%$)**: Multiplies amplitudes by $1 + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 0.03)$ to simulate motor voltage fluctuations.
2. **White Noise Floor Modulation**: Injects faint Gaussian noise ($\text{SNR} = 30\text{ dB}$) to simulate varying background ambient workshop vibrations.
3. **Speed Drift Jitter ($\pm 1.5\%$)**: Slightly perturbs harmonic center frequencies to accommodate real-world governor and slip variations.

---

## 6. Anomaly Scoring & Adaptive Thresholding

### Reconstruction Error Metric
We use Mean Squared Error (MSE) over the $B = 512$ spectral bins:

$$\mathcal{L}(x, \hat{x}) = \frac{1}{B} \sum_{i=1}^{B} \left( x_i - \hat{x}_i \right)^2$$

### Statistical Threshold Calibration
After training, evaluate the reconstruction error across a held-out set of healthy validation windows ($X_{\text{val, healthy}}$):
1. Compute error mean: $\mu_{\text{val}} = \frac{1}{M}\sum \mathcal{L}_j$
2. Compute error standard deviation: $\sigma_{\text{val}} = \text{std}(\mathcal{L}_j)$
3. Set default threshold:

$$\tau = \mu_{\text{val}} + 3.5 \cdot \sigma_{\text{val}}$$

Under Gaussian error assumptions, this guarantees a theoretical false positive rate of $< 0.05\%$.

---

## 7. Human-in-the-Loop Feedback Formula

When deployed live, technician feedback from the dashboard updates the threshold dynamically:

```python
# False Alarm feedback received from dashboard
def adjust_threshold_on_false_alarm(current_threshold, false_alarm_score, safety_margin=1.05):
    # Widen threshold just enough to envelope the safe operating state
    new_threshold = max(current_threshold, false_alarm_score * safety_margin)
    log_feedback(event="FALSE_ALARM", old_thresh=current_threshold, new_thresh=new_threshold)
    return new_threshold
```

---

## 8. Export & Deployment Contract

1. **Colab Output**:
   * Saves `autoencoder_best.pt` (FP32 weights + architecture metadata).
   * Saves `autoencoder_quantized.pt` (INT8 dynamic quantization for CPU).
2. **Raspberry Pi Deployment**:
   * Drop the `.pt` file into `detectors/saved/autoencoder.pt`.
   * Set `active: autoencoder` in `config.yaml`.
   * The live service `python -m service.monitor` loads the model and runs real-time inference seamlessly.
