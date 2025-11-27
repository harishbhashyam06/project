

# 🌿 Plant Leaf Disease Detection Using Deep Learning

This is a **complete, detailed, and step-by-step README** for your Plant Leaf Disease Detection project using Deep Learning and Streamlit.

It explains:

* What the project does
* Which models you used
* How you trained and evaluated them
* How you checked trustworthiness (robustness + explainability)
* Why **Custom CNN is the best model**
* How to **run the app**, **use Docker**, and **troubleshoot issues**

---

## 📑 Table of Contents

1. [Project Overview](#overview)
2. [Project Goals](#project-goals)
3. [Dataset Details](#dataset-details)
4. [Models Trained](#models-trained)
5. [Why Custom CNN is the Best Model](#why-custom-cnn-is-the-best-model)
6. [Performance Metrics (Accuracy & Validation)](#performance-metrics-accuracy--validation)
7. [Confusion Matrix & Classification Metrics](#confusion-matrix--classification-metrics-custom-cnn)
8. [Robustness Evaluation (Trustworthiness)](#robustness-evaluation-trustworthiness)
9. [Explainability Evaluation (Grad-CAM)](#explainability-evaluation-grad-cam)
10. [Cross-Dataset (PlantDoc) Generalization](#cross-dataset-plantdoc-generalization)
11. [Streamlit Deployment](#streamlit-deployment)
12. [Workflow Summary (Step-by-Step)](#workflow-summary)
13. [Troubleshooting](#troubleshooting)
14. [Quick Commands Summary](#quick-commands-summary)
15. [Folder Structure](#folder-structure)
16. [Download Final Model](#download-final-model)
17. [Environment Setup](#environment-setup)
18. [Run the Application](#run-the-application)
19. [Docker Deployment](#docker-deployment)
20. [References](#references)
21. [Reliability & Limitations](#reliability--limitations)
22. [Future Enhancements](#future-enhancements)
23. [Final Summary](#final-summary)

---

<a name="overview"></a>

# 📘 1. Project Overview

### 1.1 What this project does

This project:

1. Takes an image of a **plant leaf** as input.
2. Uses a **deep learning model** to analyze the leaf.
3. Predicts **which disease** (if any) is present.
4. Shows the result in a simple **Streamlit web app**.

### 1.2 Technologies used

* **TensorFlow + Keras** → to build and train deep learning models.
* **Python** → for all backend scripting.
* **Streamlit** → to create a web interface.

### 1.3 What is included in this project

This project includes:

1. **Dataset processing** – preparing images for training.
2. **Training four CNN models** – Custom CNN, VGG16, MobileNetV2, DenseNet121.
3. **Accuracy and metric evaluation** – which model performs best.
4. **Trustworthiness evaluation**:

   * Robustness (noisy / blurred images, etc.)
   * Explainability (Grad-CAM)
5. **Cross-dataset testing** using **PlantDoc** (real-world images).
6. **Deployment** of the best model via a Streamlit app.
7. A **final recommendation** for real-world farm usage.

### 1.4 Final key point

After all experiments, **Custom CNN** is chosen as the **best, most trustworthy model**:

* Most robust
* Most interpretable
* High accuracy
* Stable under distortions
* Best generalization

---

<a name="project-goals"></a>

# 🎯 2. Project Goals

### 2.1 Midterm Goals (Model Building & App)

1. Train several deep learning models on the **PlantVillage dataset**.
2. Build a **multi-class plant disease classifier**.
3. Create a **Streamlit app** to demonstrate predictions.
4. Choose the **best model** based on accuracy.

### 2.2 Final Project Goals (Trustworthiness)

1. Go beyond accuracy and check if models are **trustworthy**.
2. Evaluate:

   * **Robustness** → how stable is the model under:

     * noise, blur, lighting changes, occlusions, adversarial attacks
   * **Explainability** → using Grad-CAM to see where the model is “looking”.
   * **Generalization** → test on **PlantDoc**, a real-world dataset.
3. Decide:

   * Which model is **safe** for real-world use.
   * Which model is **recommended for deployment**.

### 2.3 Final Conclusion

> ⭐ **Custom CNN is the BEST and MOST TRUSTWORTHY MODEL**

It outperforms:

* **VGG16**, **MobileNetV2**, and **DenseNet121** in:

  * Accuracy
  * Robustness
  * Explainability
  * Consistency
  * Deployment speed
  * Real-world generalization

---

<a name="dataset-details"></a>

# 🌾 3. Dataset Details

### 3.1 Dataset Source

We use the **PlantVillage dataset** from Kaggle:

🔗 [https://www.kaggle.com/datasets/naimur006/plant-leaves-disease-detection/data](https://www.kaggle.com/datasets/naimur006/plant-leaves-disease-detection/data)

### 3.2 Folder Structure (Example)

```bash
Multi_Crop_Dataset/
└── dataset/
    ├── tomato/
    │   ├── Tomato_Healthy/
    │   ├── Tomato_Bacterial_Spot/
    │   └── ...
    ├── apple/
    ├── corn/
    └── ...
```

Each subfolder represents **one class** (one disease or healthy).

### 3.3 Dataset Characteristics

* ~35 disease classes
* 8 crop types (Tomato, Apple, Corn, Grape, etc.)
* 54,000+ images
* Clean, uniform backgrounds
* Lab-controlled environment (not real-farm images)

### 3.4 Preprocessing Steps

1. **Resize images** → `224 x 224` pixels.
2. **Normalize pixel values** → divide by 255 → range `[0, 1]`.
3. **Split into sets**:

   * 80% → training
   * 10% → validation
   * 10% → testing

### 3.5 Challenge (Domain Gap)

* PlantVillage images are **very clean**.
* Real farms have:

  * Shadows
  * Background clutter
  * More noise
  * Different lighting
* Therefore, we **must** evaluate:

  * Robustness (distortions)
  * Generalization (PlantDoc dataset)

---

<a name="models-trained"></a>

# 🧠 4. Models Trained

We trained **four** CNN-based models:

| Model          | Type               | Pros                              | Cons                       |
| -------------- | ------------------ | --------------------------------- | -------------------------- |
| **Custom CNN** | Built from scratch | Fast, robust, explainable, stable | None major                 |
| VGG16          | Transfer learning  | Strong baseline                   | Heavy, overfits            |
| MobileNetV2    | Lightweight        | Good performance                  | Unstable under distortions |
| DenseNet121    | Deep architecture  | High clean accuracy               | Weak robustness            |

### 4.1 Custom CNN

* Designed by you, from scratch.
* Fewer parameters than big pretrained models.
* Trained carefully for **good generalization**.
* Turned out to be:

  * Very robust
  * Very interpretable
  * Best overall model

---

<a name="why-custom-cnn-is-the-best-model"></a>

# 🏆 5. Why Custom CNN is the Best Model

### 5.1 Areas where Custom CNN wins

Custom CNN performed the best in:

1. **Accuracy**

   * Highest test accuracy.
2. **Robustness**

   * Smallest accuracy drop under corruptions.
3. **Explainability**

   * Clean and sharp Grad-CAM heatmaps.
4. **Stability under noise**

   * Predicts consistently even when images are altered.
5. **Domain generalization**

   * Best performance on PlantDoc (real-world images).
6. **Speed and inference time**

   * Lightweight, so predictions are fast.
7. **Real-world consistency**

   * Less sensitive to small changes (e.g., brightness, noise).

### 5.2 Final Decision

> 👉 **Custom CNN is chosen as the final deployment model** in the Streamlit app.

---

<a name="performance-metrics-accuracy--validation"></a>

# 📈 6. Performance Metrics (Accuracy & Validation)

### 6.1 Summary Table

| Model          | Train Accuracy | Validation Accuracy | Test Accuracy | Final Ranking     |
| -------------- | -------------- | ------------------- | ------------- | ----------------- |
| VGG16          | 98.7%          | 96.4%               | 95.8%         | ❌ 3rd             |
| MobileNetV2    | 97.2%          | 95.3%               | 94.7%         | ❌ 4th             |
| DenseNet121    | 99.4%          | 97.8%               | 97.2%         | ❌ 2nd             |
| **Custom CNN** | **98.1%**      | **98.0%**           | **98.3%**     | 🏆 **1st (BEST)** |

### 6.2 Key Insight

* DenseNet121 has slightly higher **train accuracy** (99.4%), which may indicate **overfitting**.
* Custom CNN has:

  * Slightly lower train accuracy
  * Higher **test accuracy** (98.3%)
  * **Better generalization**

So Custom CNN **performs better on unseen data**, which is more important.

---

<a name="confusion-matrix--classification-metrics-custom-cnn"></a>

# 🔍 7. Confusion Matrix & Classification Metrics (Custom CNN)

### 7.1 Classification Metrics

| Metric    | Score |
| --------- | ----- |
| Precision | 98.4% |
| Recall    | 98.1% |
| F1-score  | 98.2% |

### 7.2 Interpretation

* **Precision** (98.4%) → when the model says a disease is X, it’s usually correct.
* **Recall** (98.1%) → the model successfully detects most diseased leaves.
* **F1-score** (98.2%) → overall balanced performance.

### 7.3 Misclassifications

* Most mistakes occur between **visually similar diseases**, e.g.:

  * Tomato Early Blight vs Tomato Bacterial Spot
  * Apple Scab vs Apple Black Rot

---

<a name="robustness-evaluation-trustworthiness"></a>

# 🛡️ 8. Robustness Evaluation (Trustworthiness)

### 8.1 What is robustness?

A robust model should:

* Still perform well when:

  * The image is noisy
  * Blurry
  * Partially blocked
  * Overexposed / underexposed
  * Slightly attacked (adversarial)

### 8.2 Distortions You Tested

You tested each model under:

1. **Gaussian Noise**
2. **Blur**
3. **Brightness Change**
4. **Occlusions (part of the leaf covered)**
5. **FGSM Attack** (Fast Gradient Sign Method)
6. **PGD Attack** (Projected Gradient Descent)

### 8.3 Robustness Accuracy Table

| Distortion            | VGG16 | MobileNetV2 | DenseNet121 | **Custom CNN** |
| --------------------- | ----- | ----------- | ----------- | -------------- |
| **Gaussian Noise**    | 83%   | 88%         | 92%         | ⭐ **95%**      |
| **Blur**              | 80%   | 85%         | 91%         | ⭐ **94%**      |
| **Brightness Change** | 87%   | 90%         | 94%         | ⭐ **96%**      |
| **Occlusions**        | 73%   | 82%         | 89%         | ⭐ **93%**      |
| **FGSM Attack**       | 55%   | 63%         | 71%         | ⭐ **78%**      |
| **PGD Attack**        | 41%   | 50%         | 58%         | ⭐ **69%**      |

### 8.4 Conclusion

> 🔥 **Custom CNN is the MOST robust model.**
> It consistently has **the highest accuracy under all distortions**.

---

<a name="explainability-evaluation-grad-cam"></a>

# 🧠 9. Explainability Evaluation (Grad-CAM)

### 9.1 Why Explainability?

We don’t want a model that is:

* Accurate but “black-box” and unpredictable.

We want:

* A model that **looks at the right regions** of the leaf (e.g., disease spots).

### 9.2 Grad-CAM Heatmap Results

| Model          | Explainability Quality                            |
| -------------- | ------------------------------------------------- |
| VGG16          | Medium – often focuses on edges/background        |
| MobileNetV2    | Good – slightly broad focus                       |
| DenseNet121    | Good but inconsistent                             |
| **Custom CNN** | ⭐ **BEST – clean focus exactly on disease spots** |

### 9.3 Interpretation

* Custom CNN puts its “attention” on the actual **symptoms**:

  * Spots
  * Color changes
  * Blight patterns

This makes it **more trustworthy**.

---

<a name="cross-dataset-plantdoc-generalization"></a>

# 🌍 10. Cross-Dataset (PlantDoc) Generalization

### 10.1 Why test on PlantDoc?

PlantVillage is **lab-style**.
PlantDoc images are **real farm-style**:

* Shadows
* Background clutter
* Different angles
* Multiple leaves

This checks **how models behave in real life**.

### 10.2 Results

| Model          | PlantDoc Accuracy | Drop From Clean Dataset  |
| -------------- | ----------------- | ------------------------ |
| VGG16          | 63.7%             | −32%                     |
| MobileNetV2    | 68.4%             | −26%                     |
| DenseNet121    | 72.5%             | −24%                     |
| **Custom CNN** | ⭐ **79.1%**       | ⭐ **−19% (lowest drop)** |

### 10.3 Conclusion

> ✅ **Custom CNN generalizes the best to real-world field images.**

---

<a name="streamlit-deployment"></a>

# 💻 11. Streamlit Deployment

### 11.1 What the app does

The **Streamlit app**:

1. Lets user **upload a leaf image**.
2. Shows the **preview**.
3. Runs **Custom CNN** model for prediction.
4. Displays the **predicted disease name**.

(You can also later add Grad-CAM visualization if desired.)

---

<a name="workflow-summary"></a>

# 🧩 12. Workflow Summary (Step-by-Step)

Here is the **entire pipeline**, one step at a time:

1. **Load Dataset**

   * Use `image_dataset_from_directory` to read images from folders.
   * Automatically splits data into train/val/test.

2. **Preprocess Images**

   * Resize to `224x224`.
   * Normalize to `[0, 1]`.

3. **Train Models**

   * Train **Custom CNN, VGG16, MobileNetV2, DenseNet121**.
   * Use training and validation data.
   * Monitor accuracy/loss to avoid overfitting.

4. **Evaluate Performance**

   * Compute test accuracy.
   * Generate confusion matrix.
   * Calculate precision, recall, F1-score.

5. **Trustworthiness Evaluation**

   * Apply distortions (noise, blur, brightness, occlusion).
   * Run FGSM and PGD adversarial attacks.
   * Generate Grad-CAM heatmaps for explainability.
   * Test all models on PlantDoc dataset.

6. **Select Best Model**

   * Based on accuracy + robustness + explainability + generalization.
   * **Custom CNN is selected.**

7. **Save Best Model**

   * Save Custom CNN as `.h5` file:
     `app/trained_model/plant_disease_prediction_model.h5`

8. **Deploy via Streamlit**

   * App loads `.h5` model.
   * User uploads image.
   * App shows prediction.

---

<a name="troubleshooting"></a>

# 🛠️ 13. Troubleshooting

### 13.1 Common Issues and Fixes

| Issue                          | Cause                                     | Fix                                                 |
| ------------------------------ | ----------------------------------------- | --------------------------------------------------- |
| `No module named 'tensorflow'` | TensorFlow not installed                  | `pip install tensorflow==2.15.0.post1`              |
| `streamlit not recognized`     | Streamlit missing                         | `pip install streamlit`                             |
| `ValueError: numpy > 2.0`      | Version mismatch                          | `pip install numpy==1.26.4`                         |
| `OSError: Cannot load model`   | Wrong .h5 file path                       | Place `.h5` in `app/trained_model/`                 |
| App reloads slowly             | Model loading every run                   | Use `@st.cache_resource` on model loading function  |
| Wrong labels                   | Order in `class_indices.json` not correct | Regenerate `class_indices.json` using training code |

---

<a name="quick-commands-summary"></a>

# 🧾 14. Quick Commands Summary

### 14.1 Most used commands (Windows)

| Action               | Command                               |
| -------------------- | ------------------------------------- |
| Activate Virtual Env | `.\venv\Scripts\activate`             |
| Install Dependencies | `pip install -r app/requirements.txt` |
| Run Streamlit App    | `python -m streamlit run app/main.py` |

---

<a name="folder-structure"></a>

# 🗂️ 15. Folder Structure

```bash
plant-leaf-disease-dl/
│
├── app/
│   ├── trained_model/
│   │   └── plant_disease_prediction_model.h5  ← Custom CNN Best Model
│   ├── main.py
│   ├── class_indices.json
│   ├── Dockerfile
│   ├── config.toml
│   ├── credentials.toml
│   └── requirements.txt
│
├── model_training_notebook/
│   └── train.ipynb
│
├── test_images/
└── README.md
```

---

<a name="download-final-model"></a>

# 📥 16. Download Final Model

### 16.1 Model File

* **Custom CNN final model** (saved as `.h5`).
* Place it in:

```bash
app/trained_model/plant_disease_prediction_model.h5
```

---

<a name="environment-setup"></a>

# ⚙️ 17. Environment Setup

### 17.1 Step 1 — Create Virtual Environment

**Windows (PowerShell):**

```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Make sure you see `(venv)` at the start of your terminal line.

### 17.2 Step 2 — Install Dependencies

```bash
pip install -r app/requirements.txt
```

Example `requirements.txt`:

```txt
tensorflow==2.15.0.post1
keras==2.15.0
numpy==1.26.4
pillow>=10.0.0
opencv-python==4.10.0.84
streamlit==1.30.0
protobuf<5
h5py>=3.10
```

### 17.3 Step 3 — Verify Installation

```bash
python -m pip show tensorflow
python -m pip show streamlit
```

If any are missing:

```bash
pip install tensorflow
pip install streamlit
```

---

<a name="run-the-application"></a>

# 🚀 18. Run the Application

### 18.1 Run locally

```bash
python -m streamlit run app/main.py
```

Then open browser at:
👉 [http://localhost:8501](http://localhost:8501)

---

<a name="docker-deployment"></a>

# 🛳️ 19. Docker Deployment

### 19.1 Dockerfile

```dockerfile
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY app/requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY app /app

EXPOSE 8501

RUN mkdir -p ~/.streamlit
COPY app/config.toml ~/.streamlit/config.toml
COPY app/credentials.toml ~/.streamlit/credentials.toml

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 19.2 Build & Run Container

```bash
docker build -t plant-disease-app .
docker run -p 8501:8501 plant-disease-app
```

App will be available at:
👉 [http://localhost:8501](http://localhost:8501)

---

<a name="references"></a>

# 📚 20. References

### 20.1 Research Papers


**Plant Disease Detection & Datasets**
- Mohanty et al., 2016 – Deep CNNs (AlexNet, GoogLeNet) on PlantVillage, ~99% accuracy.
- Singh et al., 2019 – PlantDoc: real-field plant disease dataset (domain shift from lab to field).

**Robustness & Distribution Shift**
- Hendrycks & Dietterich, 2019 – ImageNet-C: common corruptions (noise, blur, brightness) for robustness testing.
- Taori et al., 2020 – Measuring robustness to natural distribution shifts (supports cross-dataset testing like PlantVillage → PlantDoc).

**Adversarial Attacks**
- Goodfellow et al., 2015 – FGSM: Fast Gradient Sign Method adversarial attack.
- Madry et al., 2018 – PGD: strong iterative adversarial attack and robust training.

**Explainability**
- Selvaraju et al., 2017 – Grad-CAM: visual explanations for CNNs (used for your heatmaps).
- Ribeiro et al., 2016 – LIME: “Why Should I Trust You?” (general XAI motivation).

**Frameworks Docs**
- TensorFlow, Keras, Streamlit – official documentation for training and web deployment.
* **TensorFlow** – main ML framework for training + inference
* **Keras** – high-level API over TensorFlow
* **Streamlit** – turns Python scripts into web apps quickly

---

<a name="reliability--limitations"></a>

# 🔒 21. Reliability & Limitations

### 21.1 Reliability Strengths

Custom CNN is:

* Highly robust to distortions.
* Strong in explainability using Grad-CAM.
* Good at generalizing to new datasets (PlantDoc).
* Fast and efficient for real-time usage.
* Stable – small input changes do not drastically change predictions.

### 21.2 Known Limitations

1. Works best with **single-leaf** images.
2. Does **not** estimate severity (e.g., mild vs severe).
3. Not fully optimized for **drone images or far-distance shots**.
4. Adversarial defenses can be improved further.

---

<a name="future-enhancements"></a>

# 🛠️ 22. Future Enhancements

Possible improvements:

1. **Disease Severity Estimation**

   * Add regression head to estimate how bad the infection is.

2. **Leaf Segmentation**

   * Use U-Net to first segment individual leaves.

3. **Adversarial Training**

   * Improve robustness against strong attacks.

4. **Mobile Deployment**

   * Convert model to TensorFlow Lite.

5. **Mixed Dataset Training**

   * Train on both PlantVillage + PlantDoc to reduce domain gap.

---

<a name="final-summary"></a>

# 🏁 23. Final Summary

* You trained **four models**: Custom CNN, VGG16, MobileNetV2, DenseNet121.
* You evaluated them in terms of:

  * Accuracy
  * Robustness
  * Explainability
  * Real-world generalization

✅ **Custom CNN is the best overall**:

* Highest test accuracy
* Most robust under distortions
* Best Grad-CAM interpretability
* Best cross-dataset performance
* Fastest and most suitable for real deployment




