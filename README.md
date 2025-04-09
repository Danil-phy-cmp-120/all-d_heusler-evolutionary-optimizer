# DFT Active Learning Optimizer

This project provides a framework for optimizing material structures using Density Functional Theory (DFT) guided by active learning and genetic algorithms.

## 🧠 Overview

The goal is to accelerate DFT-based optimization by using trained surrogate Random Forest models and efficient sampling strategies. The project includes:
- Sampling scripts for generating DFT input data
- Active learning loops for structure-property optimization
- Trained machine learning models for property prediction
- A genetic algorithm-based optimization engine

## 🗂 Project Structure

```
dft-active-learning-optimizer/
│
├── 1.1.sampler_for_dft/           # Sampling tools and models
│   ├── sampler_dft_random.py
│   ├── sampler_dft_uniform.py
│   ├── model_*.pickle             # Trained surrogate models
│   └── initial_poscars/           # Example POSCARs
│
├── trained_models/                # Pre-trained ML models for DFT properties
│
├── active_learning_*.py          # Scripts for various optimization tasks:
│   ├── mag_aust.py, mag_mart.py, tetr.py, vol.py
│
├── optimizator_GA.py             # Genetic Algorithm optimizer
├── df.csv                        # Dataset
└── LICENSE
```

## ⚙️ Installation

Recommended: Python 3.8+ with the following packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

Additional tools for DFT workflows (e.g., VASP) may be required for full functionality.

## 🚀 Usage

You can start an active learning loop by running one of the scripts, for example:

```bash
python active_learning_tetr.py
```

To use the genetic algorithm optimizer:

```bash
python optimizator_GA.py
```

Make sure all models and input data are in place.

## 📄 License

See the `LICENSE` file for license details.
