
# 🌿 Plant Leaf Disease Detection Using Deep Learning

This is a **complete, detailed, and step-by-step README** for our Plant Leaf Disease Detection project using Deep Learning and Streamlit.

It explains:

- What the project does  
- Which models we used  
- How we trained and evaluated them  
- How we checked **trustworthiness**, with a primary focus on **Reliability and Robustness** (course requirement)  
- How we used **Grad-CAM** as a supporting explainability tool  
- Why **Custom CNN is the best model**  
- How to **run the app**, **run robustness & attack evaluations**, **use Docker**, and **troubleshoot issues**

---

## 📑 Table of Contents

1. [Project Overview](#overview)  
2. [Project Goals](#project-goals)  
3. [Dataset Details](#dataset-details)  
4. [Environment Setup](#environment-setup)  
5. [Folder Structure](#folder-structure)  
6. [Models Trained](#models-trained)  
7. [Performance Metrics (Accuracy & Validation)](#performance-metrics-accuracy--validation)  
8. [Confusion Matrix & Classification Metrics](#confusion-matrix--classification-metrics-custom-cnn)  
9. [Robustness Evaluation (Trustworthiness)](#robustness-evaluation-trustworthiness)  
10. [Explainability Evaluation (Grad-CAM)](#explainability-evaluation-grad-cam)  
11. [Cross-Dataset (PlantDoc) Generalization](#cross-dataset-plantdoc-generalization)  
12. [Why Custom CNN is the Best Model](#why-custom-cnn-is-the-best-model)  
13. [Reliability & Limitations](#reliability--limitations)  
14. [Download Final Model](#download-final-model)  
15. [Streamlit Deployment](#streamlit-deployment)  
16. [Run the Application](#run-the-application)  
17. [Docker Deployment](#docker-deployment)  
18. [Test the Model](#test-the-model)  
19. [Workflow Summary (Step-by-Step)](#workflow-summary)  
20. [Quick Commands Summary](#quick-commands-summary)  
21. [Troubleshooting](#troubleshooting)  
22. [Future Enhancements](#future-enhancements)  
23. [References](#references)  
24. [Final Summary](#final-summary)

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

- **TensorFlow + Keras** → to build and train deep learning models.  
- **Python** → for all backend scripting.  
- **Streamlit** → to create a web interface.  

### 1.3 What is included in this project

This project includes:

1. **Dataset processing** – preparing images for training.  
2. **Training four CNN models** – Custom CNN, VGG16, MobileNetV2, DenseNet121.  
3. **Accuracy and metric evaluation** – which model performs best.  
4. **Trustworthiness evaluation** with a focus on **Reliability and Robustness**:
   - Common distortions (noise, blur, brightness changes, occlusions)  
   - Adversarial attacks (FGSM, PGD)  
   - Cross-dataset testing using **PlantDoc** (real-world images)  
   - **Grad-CAM** as a supporting tool to check if the model looks at the right leaf regions  
5. **Deployment** of the best model via a Streamlit app.  
6. A **final recommendation** for real-world farm usage.

### 1.4 Final key point

After all experiments, **Custom CNN** is chosen as the **best, most trustworthy model**:

- Most robust  
- Most interpretable (via Grad-CAM)  
- High accuracy  
- Stable under distortions and attacks  
- Best generalization to real-world data  

### 1.5 Team Members & Contributions

This project was completed by a team of **three members**:


- **Member 1 – \<Harish Bhashyam\>**  
  - Collected and organized the **PlantVillage** dataset.  
  - Implemented **data preprocessing** (resizing, normalization, train/val/test splits).  
  - Designed and trained the **Custom CNN** model.  
  - Helped prepare the **classification metrics** (precision, recall, F1-score).
  - Built the **Streamlit app**, Dockerfile, and wrote the **README documentation**.

- **Member 2 – \<Vineeth  Godishala\>**  
  - Implemented and fine-tuned **transfer learning models**: VGG16, MobileNetV2, DenseNet121.  
  - Ran **baseline accuracy evaluations** and created the **performance comparison tables**.  
  - Performed **cross-dataset experiments** on **PlantDoc** and analyzed generalization.  
  - Contributed to writing the **model comparison and final decision** sections.

- **Member 3 – \<Sachith Reddy Mekala\>**  
  - Implemented **robustness evaluations**: noise, blur, brightness, occlusions.  
  - Implemented **FGSM and PGD adversarial attacks** and summarized robustness tables.  
  - Implemented **Grad-CAM explainability** for all four models.  


All three members jointly discussed design choices, analyzed results, and finalized the conclusion that **Custom CNN** is the most reliable and robust model for deployment.

---

<a name="project-goals"></a>

# 🎯 2. Project Goals

### 2.1 Midterm Goals (Model Building & App)

1. Train several deep learning models on the **PlantVillage dataset**.  
2. Build a **multi-class plant disease classifier**.  
3. Create a **Streamlit app** to demonstrate predictions.  
4. Choose the **best model** based on accuracy.

### 2.2 Final Project Goals (Trustworthiness)

For the final project, we **go beyond accuracy** and check if the models are **trustworthy**.

We evaluate:

- **Robustness / Reliability** → how stable is the model under:
  - Noise  
  - Blur  
  - Lighting changes  
  - Occlusions  
  - Adversarial attacks (FGSM, PGD)  
- **Generalization** → test on **PlantDoc**, a real-world dataset with natural farm images.  
- **Explainability (supporting)** → use Grad-CAM to see where the model is “looking” and verify that it focuses on disease regions.

We then decide:

- Which model is **safe** for real-world use.  
- Which model is **recommended for deployment** in the Streamlit app.

### 2.3 Final Conclusion

> ⭐ **Custom CNN is the BEST and MOST TRUSTWORTHY MODEL**

It outperforms **VGG16**, **MobileNetV2**, and **DenseNet121** in:

- Accuracy  
- Robustness (distortions + attacks)  
- Consistency  
- Real-world generalization (PlantDoc)  
- Inference speed and deployment readiness  

### 2.4 Trustworthiness Focus (Final Project Requirement)

According to the course requirement, this final project evaluates the AI application based on **one specific aspect of trustworthiness**:

> ✅ **Primary Trustworthiness Aspect: Reliability and Robustness**

In this project, we:

- Systematically test how the models behave under **corruptions** (noise, blur, brightness changes, occlusions).  
- Evaluate against **adversarial attacks** (FGSM, PGD).  
- Measure performance under **dataset shift** (PlantVillage → PlantDoc).  

**Grad-CAM explainability** is used as a **supporting tool**, not the main aspect, to verify that the model’s decisions are reasonable and consistent with disease regions.

---

<a name="dataset-details"></a>

# 🌾 3. Dataset Details

### 3.1 Dataset Source

We use the **PlantVillage dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/naimur006/plant-leaves-disease-detection/data

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
````

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

PlantVillage images are **very clean**, but real farms have:

* Shadows
* Background clutter
* More noise
* Different lighting

Therefore, we **must** evaluate:

* **Robustness** (under distortions and attacks)
* **Generalization** (PlantVillage → PlantDoc dataset)

---

<a name="environment-setup"></a>

# ⚙️ 4. Environment Setup

### 4.1 Step 1 — Create Virtual Environment

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

### 4.2 Step 2 — Install Dependencies

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

### 4.3 Step 3 — Verify Installation

```bash
python -m pip show tensorflow
python -m pip show streamlit
```

If any are missing:

```bash
pip install tensorflow
pip install streamlit
```

### 4.4 How to Run the Trustworthiness Evaluation (Robustness, Attacks, Grad-CAM)

All robustness, adversarial, and explainability experiments are implemented in:

```text
model_training_notebook/train.ipynb
```

To reproduce the trustworthiness evaluation:

1. **Activate the virtual environment** (see above).

2. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook model_training_notebook/train.ipynb
   ```

3. Inside the notebook, run the cells in these sections (or similarly named blocks):

   * **Robustness Tests**

     * Applies Gaussian noise, blur, brightness changes, and occlusions.
   * **Adversarial Attacks**

     * Implements **FGSM** and **PGD** attacks on test images.
   * **Grad-CAM Explainability**

     * Generates heatmaps for all four models.
   * **PlantDoc Cross-Dataset Evaluation**

     * Evaluates each model on the PlantDoc dataset and computes accuracy / drop.

4. The notebook will output:

   * Robustness accuracy tables (used in Section 9)
   * Grad-CAM visualizations (used in Section 10)
   * Cross-dataset results (used in Section 11)

These experiments directly support our **Reliability & Robustness** evaluation for the final project.

---

<a name="folder-structure"></a>

# 🗂️ 5. Folder Structure

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

<a name="models-trained"></a>

# 🧠 6. Models Trained

The best-performing model (**Custom CNN**) was selected for deployment.

📥 **Download Model**: [Google Drive Link](https://drive.google.com/file/d/13uQJa-Bq1sEv2ai_XdrmvrJgWWba7v-7/view)

After download, place it here:

```
app/trained_model/plant_disease_prediction_model.h5
```

We trained **four** CNN-based models:

| Model          | Type               | Pros                              | Cons                       |
| -------------- | ------------------ | --------------------------------- | -------------------------- |
| **Custom CNN** | Built from scratch | Fast, robust, explainable, stable | None major                 |
| VGG16          | Transfer learning  | Strong baseline                   | Heavy, overfits            |
| MobileNetV2    | Lightweight        | Good performance                  | Unstable under distortions |
| DenseNet121    | Deep architecture  | High clean accuracy               | Weak robustness            |

### 6.1 Custom CNN

* Designed from scratch.
* Fewer parameters than big pretrained models.
* Trained carefully for **good generalization**.
* Turned out to be:

  * Very robust
  * Very interpretable
  * Best overall model

---

<a name="performance-metrics-accuracy--validation"></a>

# 📈 7. Performance Metrics (Accuracy & Validation)

### 7.1 Summary Table

| Model          | Train Accuracy | Validation Accuracy | Test Accuracy | Final Ranking     |
| -------------- | -------------- | ------------------- | ------------- | ----------------- |
| VGG16          | 98.7%          | 96.4%               | 95.8%         | 3rd               |
| MobileNetV2    | 97.2%          | 95.3%               | 94.7%         | 4th               |
| DenseNet121    | 99.4%          | 97.8%               | 97.2%         | 2nd               |
| **Custom CNN** | **98.1%**      | **98.0%**           | **98.3%**     | 🏆 **1st (BEST)** |

### 7.2 Key Insight

* DenseNet121 has slightly higher **train accuracy** (99.4%), which may indicate **overfitting**.
* Custom CNN has:

  * Slightly lower train accuracy
  * Higher **test accuracy** (98.3%)
  * **Better generalization**

So Custom CNN **performs better on unseen data**, which is more important.

---

<a name="confusion-matrix--classification-metrics-custom-cnn"></a>

# 🔍 8. Confusion Matrix & Classification Metrics (Custom CNN)

### 8.1 Classification Metrics

| Metric    | Score |
| --------- | ----- |
| Precision | 98.4% |
| Recall    | 98.1% |
| F1-score  | 98.2% |

### 8.2 Interpretation

* **Precision (98.4%)** → when the model says a disease is X, it’s usually correct.
* **Recall (98.1%)** → the model successfully detects most diseased leaves.
* **F1-score (98.2%)** → overall balanced performance.

### 8.3 Misclassifications

Most mistakes occur between **visually similar diseases**, e.g.:

* Tomato Early Blight vs Tomato Bacterial Spot
* Apple Scab vs Apple Black Rot

---

<a name="robustness-evaluation-trustworthiness"></a>

# 🛡️ 9. Robustness Evaluation (Trustworthiness)

### 9.1 What is robustness?

A robust model should still perform well when:

* The image is **noisy**
* **Blurry**
* **Partially blocked**
* **Overexposed / underexposed**
* Slightly attacked by **adversarial perturbations**

### 9.2 Distortions We Tested

We tested each model under:

1. **Gaussian Noise**
2. **Blur**
3. **Brightness Change**
4. **Occlusions** (part of the leaf covered)
5. **FGSM Attack** (Fast Gradient Sign Method)
6. **PGD Attack** (Projected Gradient Descent)

### 9.3 Robustness Accuracy Table

| Distortion            | VGG16 | MobileNetV2 | DenseNet121 | **Custom CNN** |
| --------------------- | ----- | ----------- | ----------- | -------------- |
| **Gaussian Noise**    | 83%   | 88%         | 92%         | ⭐ **95%**      |
| **Blur**              | 80%   | 85%         | 91%         | ⭐ **94%**      |
| **Brightness Change** | 87%   | 90%         | 94%         | ⭐ **96%**      |
| **Occlusions**        | 73%   | 82%         | 89%         | ⭐ **93%**      |
| **FGSM Attack**       | 55%   | 63%         | 71%         | ⭐ **78%**      |
| **PGD Attack**        | 41%   | 50%         | 58%         | ⭐ **69%**      |

### 9.4 Conclusion

> 🔥 **Custom CNN is the MOST robust model.**
> It consistently has **the highest accuracy under all distortions and attacks**.

This directly supports our chosen aspect of trustworthiness: **Reliability and Robustness**.

---

<a name="explainability-evaluation-grad-cam"></a>

# 🧠 10. Explainability Evaluation (Grad-CAM)

### 10.1 Why Explainability?

We don’t want a model that is:

* Accurate but **“black-box”** and unpredictable.

We want:

* A model that **looks at the right regions** of the leaf (e.g., disease spots) when making predictions.

### 10.2 Grad-CAM Heatmap Results

| Model          | Explainability Quality                            |
| -------------- | ------------------------------------------------- |
| VGG16          | Medium – often focuses on edges/background        |
| MobileNetV2    | Good – slightly broad focus                       |
| DenseNet121    | Good but inconsistent                             |
| **Custom CNN** | ⭐ **BEST – clean focus exactly on disease spots** |

### 10.3 Interpretation

Custom CNN puts its “attention” on the actual **symptoms**:

* Spots
* Color changes
* Blight patterns

This makes it **more trustworthy**, and it supports our robustness findings by showing the model is using meaningful visual features.

---

<a name="cross-dataset-plantdoc-generalization"></a>

# 🌍 11. Cross-Dataset (PlantDoc) Generalization

### 11.1 Why test on PlantDoc?

PlantVillage is **lab-style**, while PlantDoc images are **real farm-style**:

* Shadows
* Background clutter
* Different angles
* Multiple leaves

This checks **how models behave in real life** (distribution shift).

### 11.2 Results

| Model          | PlantDoc Accuracy | Drop From Clean Dataset  |
| -------------- | ----------------- | ------------------------ |
| VGG16          | 63.7%             | −32%                     |
| MobileNetV2    | 68.4%             | −26%                     |
| DenseNet121    | 72.5%             | −24%                     |
| **Custom CNN** | ⭐ **79.1%**       | ⭐ **−19% (lowest drop)** |

### 11.3 Conclusion

> ✅ **Custom CNN generalizes the best to real-world field images.**
> Its performance degrades the least when moving from lab images to real farm images, which is critical for reliability.

---

<a name="why-custom-cnn-is-the-best-model"></a>

# 🏆 12. Why Custom CNN is the Best Model

### 12.1 Areas where Custom CNN wins

Custom CNN performed the best in:

1. **Accuracy**

   * Highest test accuracy.

2. **Robustness**

   * Smallest accuracy drop under distortions and adversarial attacks.

3. **Explainability**

   * Clean and sharp Grad-CAM heatmaps focusing on disease regions.

4. **Stability under noise**

   * Predicts consistently even when images are altered.

5. **Domain generalization**

   * Best performance on PlantDoc (real-world images).

6. **Speed and inference time**

   * Lightweight, so predictions are fast.

7. **Real-world consistency**

   * Less sensitive to small changes (e.g., brightness, noise).

### 12.2 Final Decision

> 👉 **Custom CNN is chosen as the final deployment model** in the Streamlit app.

---

<a name="reliability--limitations"></a>

# 🔒 13. Reliability & Limitations

### 13.1 Reliability Strengths

Custom CNN is:

* Highly robust to distortions and adversarial attacks.
* Strong in explainability using Grad-CAM.
* Good at generalizing to new datasets (PlantDoc).
* Fast and efficient for real-time usage.
* Stable – small input changes do not drastically change predictions.

### 13.2 Known Limitations

1. Works best with **single-leaf** images.
2. Does **not** estimate severity (e.g., mild vs severe).
3. Not fully optimized for **drone images or far-distance shots**.
4. Adversarial defenses (robust training) can be improved further.

---

<a name="download-final-model"></a>

# 📥 14. Download Final Model

### 14.1 Model File

* **Custom CNN final model** (saved as `.h5`).
* Place it in:

```bash
app/trained_model/plant_disease_prediction_model.h5
```

---

<a name="streamlit-deployment"></a>

# 💻 15. Streamlit Deployment

### 15.1 What the app does

The **Streamlit app**:

1. Lets the user **upload a leaf image**.
2. Shows the **preview**.
3. Runs the **Custom CNN** model for prediction.
4. Displays the **predicted disease name**.

---

<a name="run-the-application"></a>

# 🚀 16. Run the Application

### 16.1 Run locally

```bash
python -m streamlit run app/main.py
```

Then open the browser at:
👉 [http://localhost:8501](http://localhost:8501)

---

<a name="docker-deployment"></a>

# 🛳️ 17. Docker Deployment

### 17.1 Dockerfile

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

### 17.2 Build & Run Container

```bash
docker build -t plant-disease-app .
docker run -p 8501:8501 plant-disease-app
```

App will be available at:
👉 [http://localhost:8501](http://localhost:8501)

---

<a name="test-the-model"></a>

# 🧪 18. Test the Model

You can verify the model’s predictions using sample test images.

### 18.1 Test Images Folder

```bash
test_images/
├── test_apple_black_rot.JPG
├── test_blueberry_healthy.jpg
└── test_potato_early_blight.jpg
```

### 18.2 Steps to Test

1. Run the Streamlit app.
2. In the browser, click **Browse files** / **Upload an Image**.
3. Choose any image from the `test_images/` folder.
4. Click the **Classify** button.
5. The app will show the predicted disease.

### 18.3 Example Output

Example prediction:

```text
Prediction: Tomato__Tomato_Yellow_Leaf_Curl_Virus
Confidence: 0.9873
```

Example UI Screenshot (replace with your actual GitHub image path):

```markdown
![App Output](https://github.com/harishbhashyam06/AI-Plant-Leaf-Disease-Detection-Using-Deep-Learning/blob/main/test_images/output.png)
```

---

<a name="workflow-summary"></a>

# 🧩 19. Workflow Summary (Step-by-Step)

Here is the **entire pipeline**, one step at a time:

1. **Load Dataset**

   * Use `image_dataset_from_directory` to read images from folders.
   * Automatically splits data into train/val/test.

2. **Preprocess Images**

   * Resize images to `224x224`.
   * Normalize pixel values to `[0, 1]`.

3. **Train Models**

   * Train **Custom CNN, VGG16, MobileNetV2, DenseNet121**.
   * Use training and validation datasets.
   * Monitor accuracy/loss to avoid overfitting.

4. **Evaluate Performance**

   * Compute test accuracy.
   * Generate confusion matrix.
   * Calculate precision, recall, and F1-score.

5. **Trustworthiness Evaluation (Reliability & Robustness)**

   * Apply distortions (noise, blur, brightness, occlusion).
   * Run FGSM and PGD adversarial attacks.
   * Generate Grad-CAM heatmaps for explainability.
   * Test all models on PlantDoc dataset.

6. **Select Best Model**

   * Consider accuracy + robustness + explainability + generalization.
   * **Custom CNN is selected as the final model.**

7. **Save Best Model**

   * Save Custom CNN as `.h5` file:
     `app/trained_model/plant_disease_prediction_model.h5`

8. **Deploy via Streamlit**

   * The app loads the `.h5` model.
   * User uploads image.
   * App shows prediction and can be extended to show Grad-CAM.

---

<a name="quick-commands-summary"></a>

# 🧾 20. Quick Commands Summary

### 20.1 Most used commands (Windows)

| Action               | Command                               |
| -------------------- | ------------------------------------- |
| Activate Virtual Env | `.\venv\Scripts\activate`             |
| Install Dependencies | `pip install -r app/requirements.txt` |
| Run Streamlit App    | `python -m streamlit run app/main.py` |

---

<a name="troubleshooting"></a>

# 🛠️ 21. Troubleshooting

### 21.1 Common Issues and Fixes

| Issue                          | Cause                                   | Fix                                                   |
| ------------------------------ | --------------------------------------- | ----------------------------------------------------- |
| `No module named 'tensorflow'` | TensorFlow not installed                | `pip install tensorflow==2.15.0.post1`                |
| `streamlit not recognized`     | Streamlit missing                       | `pip install streamlit`                               |
| `ValueError: numpy > 2.0`      | Version mismatch                        | `pip install numpy==1.26.4`                           |
| `OSError: Cannot load model`   | Wrong `.h5` file path                   | Place `.h5` in `app/trained_model/`                   |
| App reloads slowly             | Model loading on every run              | Use `@st.cache_resource` for model loading function   |
| Wrong labels                   | `class_indices.json` not matching model | Regenerate `class_indices.json` using training script |

---

<a name="future-enhancements"></a>

# 🛠️ 22. Future Enhancements

Possible improvements:

1. **Disease Severity Estimation**

   * Add regression head to estimate how bad the infection is.

2. **Leaf Segmentation**

   * Use U-Net to first segment individual leaves.

3. **Adversarial Training**

   * Train the model with adversarial examples to improve robustness.

4. **Mobile Deployment**

   * Convert model to TensorFlow Lite.

5. **Mixed Dataset Training**

   * Train on both PlantVillage + PlantDoc to reduce domain gap.

---

<a name="references"></a>

# 📚 23. References

### 23.1 Research Papers (Midterm + Final)

**Plant Disease Detection & Datasets**

* Mohanty et al., 2016 – Deep CNNs (AlexNet, GoogLeNet) on PlantVillage, ~99% accuracy.
* Singh et al., 2019 – PlantDoc: real-field plant disease dataset (domain shift from lab to field).

**Robustness & Distribution Shift**

* Hendrycks & Dietterich, 2019 – ImageNet-C: common corruptions (noise, blur, brightness) for robustness testing.
* Taori et al., 2020 – Measuring robustness to natural distribution shifts (supports cross-dataset testing like PlantVillage → PlantDoc).

**Adversarial Attacks**

* Goodfellow et al., 2015 – FGSM: Fast Gradient Sign Method adversarial attack.
* Madry et al., 2018 – PGD: strong iterative adversarial attack and robust training.

**Explainability**

* Selvaraju et al., 2017 – Grad-CAM: visual explanations for CNNs (used for our heatmaps).
* Ribeiro et al., 2016 – LIME: “Why Should I Trust You?” (general XAI motivation).

### 23.2 Framework Documentation

* **TensorFlow** – main ML framework for training + inference
* **Keras** – high-level API over TensorFlow
* **Streamlit** – turns Python scripts into web apps quickly

(Official docs: tensorflow.org, keras.io, docs.streamlit.io)

### 23.3 Code Sources & Attribution

* All training, evaluation, robustness, and Streamlit app code in this repository was primarily written by **our project team**.
* We referred to the official **TensorFlow/Keras** and **Streamlit** documentation for API usage (model building, data loading, and app setup).
* The **FGSM** and **PGD** adversarial attack methods are based on the concepts from:

  * Goodfellow et al., 2015 (FGSM)
  * Madry et al., 2018 (PGD)
* The **Grad-CAM** implementation follows the method proposed by **Selvaraju et al., 2017**.
* Any small code snippets adapted from online tutorials or documentation pages are acknowledged with comments inside the corresponding notebook cells or scripts.

---

<a name="final-summary"></a>

# 🏁 24. Final Summary

We trained **four models**: Custom CNN, VGG16, MobileNetV2, DenseNet121.

We evaluated them in terms of:

* Accuracy
* **Reliability & Robustness** (distortions + adversarial attacks)
* Explainability (supporting, via Grad-CAM)
* Real-world generalization (PlantDoc)

✅ **Custom CNN is the best overall**:

* Highest test accuracy
* Most robust under distortions and attacks
* Best Grad-CAM interpretability
* Best cross-dataset performance
* Fastest and most suitable for real deployment

---

✅ **Run Summary**

```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
pip install -r app/requirements.txt
python -m streamlit run app/main.py
```



