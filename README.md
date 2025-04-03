# 🛑 Deprecated – See New Version: [AirSeg](https://github.com/chetana348/AirSeg-Learnable-Interconnected-Attention-Framework-for-Robust-Airway-Segmentation)

---

## ⚠️ Notice

This repository contains an **older version** of our airway segmentation model and is now **deprecated**.

For the latest version with improved performance, better generalization, and a learnable attention mechanism, please refer to our new repository:

👉 **[AirSeg: Learnable Interconnected Attention Framework for Robust Airway Segmentation](https://github.com/chetana348/AirSeg-Learnable-Interconnected-Attention-Framework-for-Robust-Airway-Segmentation)**

---

## ✅ Why Keep This Repo?

While deprecated, this model:
- Remains fully functional
- Outperforms baseline models for lung segmentation
- Can serve as a **lightweight baseline**
- Offers a great **starting point** for airway segmentation tasks or model prototyping

> 💡 Ideal for researchers who want to understand the foundations of the newer AirSeg architecture or build simpler extensions.

---

## 📌 Overview

This model is designed for **segmentation of lungs and airways** from 3D chest CT scans.

Unlike traditional U-Net-based models, this approach is particularly effective at:

- Capturing **tiny, fragmented, or discontinuous airway branches**
- Preserving **fine structural details** in both proximal and distal airways
- Providing a more accurate and complete segmentation of the airway tree

> 🫁 Ideal for applications in **pulmonary imaging**, **preoperative planning**, and **airway disease analysis**

---

## ⚙️ Installation

To set up the environment for this project, use the provided `environment.yml` file.

### 🔧 Step-by-Step

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate airway_segmentation_env
   ```

3. Ensure you have a compatible **GPU setup** for TensorFlow:
   - CUDA and cuDNN should match your TensorFlow GPU version
   - Recommended TensorFlow: `tensorflow-gpu >= 2.x`

> ⚠️ This project is optimized for **TensorFlow GPU**. Using only CPU may result in significantly slower inference and training times.

---

## 📚 Dataset & Model Weights

This model was originally trained on **proprietary chest CT datasets**. While the dataset is not publicly available, the model has shown strong **generalization to other CT datasets** with similar preprocessing.

### 🔗 Pretrained Weights

You can download the pretrained model weights from the following Google Drive link:

📦 [Model Weights – Google Drive](https://drive.google.com/drive/folders/1H1QUAC9UnkFKWtnHB_wgukHdUJREbNCa)

These weights can be used directly for inference or fine-tuning on custom datasets.

---

## 🧪 Preprocessing Pipeline

To prepare your CT scans for training or inference, follow these steps:

1. **Resize** all slices to `128 × 128` pixels  
2. **Slice** 3D volumes into individual 2D axial slices  
3. **Normalize** intensity values to the range `[0, 1]` or `[0, 255]` depending on the model input expectations

> ⚠️ Ensure consistent spacing and alignment across datasets for optimal performance.

---

## 📁 Repository Structure

Below is an overview of the key files and folders in this repository:

```
├── Networks/                         # Model architecture scripts and building blocks
├── saved/                            # Folder containing pretrained model weights
│
├── 1. Pre-Processing.ipynb           # Resize, slice, and normalize CT images
├── 2. Train_Prediction_Evaluation.ipynb  # Full pipeline for training, inference, and evaluation
├── Only predictions.ipynb           # Inference-only notebook using pretrained weights
│
├── Functions_Airways1.ipynb         # Postprocessing and visualization for airway segmentation
├── Functions_Lungs1.ipynb           # Postprocessing and visualization for lung segmentation
│
├── Data_Gen_2D.py                   # Data generator for 2D image slices
├── switchnorm.py                    # Custom normalization layer (SwitchNorm)
│
├── environment.yml                  # Conda environment file with all dependencies
├── .gitattributes                   # Git configuration
├── SOP.pptx                         # Presentation showing step-by step to run
├── README.md                        # You are here 📄
```

> 📦 **Note:** Pretrained model weights are located in the `saved/` directory and can be loaded directly in the notebooks.

---

## 🧪 Try It in Google Colab

Want to try the model without setting up your local environment?

You can test the full pipeline — including preprocessing, prediction, and visualization — using our **Colab notebook**:

🔗 **[Run in Google Colab](https://drive.google.com/drive/folders/1SBL6bOjsyhwK8Ib5fgrQa4Zr8peVuYXn?usp=sharing)**

> 📦 The notebook is preconfigured to load the pretrained weights and sample data.  
> Make sure to mount your Google Drive when prompted and follow the step-by-step cells.

---

## 📄 License

This project is licensed under the **MIT License**.

