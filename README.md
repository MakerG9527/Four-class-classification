# Four-class Classification

[![Paper](https://img.shields.io/badge/DOI-10.1016%2Fj.apsusc.2024.161276-blue)](https://doi.org/10.1016/j.apsusc.2024.161276)

This repository contains the code for our paper published in **Applied Surface Science**.

> **Citation**: If you use this code in your research, please cite our paper:
> - DOI: https://doi.org/10.1016/j.apsusc.2024.161276

---

## 📊 Results Preview

View the comprehensive results: **[merged.pdf](./merged.pdf)**

<p align="center">
  <a href="./merged.pdf">
    <img src="https://img.shields.io/badge/PDF-View%20Results-red?style=for-the-badge&logo=adobeacrobatreader" alt="View PDF">
  </a>
</p>

---

## 🚀 Quick Start

### 1. Divide the data with different step lengths and create the dataset

```bash
python ./grouping.py
```

> Origin data in folder `Datas` will be divided into a new folder called `Step_XXX`.

### 2. Run different models to classify the dataset

```bash
python ./models/MLP.py
python ./models/CNN.py
python ./models/RNN.py
python ./models/LSTM.py
python ./models/Transformer.py
```

> Running results will be saved in folder `logs`.

### 3. Plot the results

```bash
python ./merged.py
```

> Precision, Recall, F1 Score, mAP, and Loss in different methods will be shown in `merged.pdf`.

---

## 📁 Repository Structure

```
.
├── Datas/              # Original dataset
├── models/             # Model implementations (MLP, CNN, RNN, LSTM, Transformer)
├── Step_*/             # Data divided with different step lengths
├── logs/               # Training logs
├── grouping.py         # Data preprocessing script
├── merged.py           # Results visualization script
├── draw_loss.py        # Loss plotting script
└── merged.pdf          # Comprehensive results visualization
```

---

## 📝 Paper Information

- **Journal**: Applied Surface Science
- **DOI**: [10.1016/j.apsusc.2024.161276](https://doi.org/10.1016/j.apsusc.2024.161276)
- **Title**: See the paper for full details

---

## 🔬 Models Implemented

- **MLP** - Multi-Layer Perceptron
- **CNN** - Convolutional Neural Network
- **RNN** - Recurrent Neural Network
- **LSTM** - Long Short-Term Memory
- **Transformer** - Transformer Architecture

---

## 📈 Results

The classification results including Precision, Recall, F1 Score, mAP, and Loss for different methods are visualized in `merged.pdf`.

<p align="center">
  <a href="./merged.pdf">
    <b>📄 Click here to view the full results (merged.pdf)</b>
  </a>
</p>
