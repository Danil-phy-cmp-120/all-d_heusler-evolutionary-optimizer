# Genetic Algorithm Optimizer

This script implements a **Genetic Algorithm (GA)** for optimizing material configurations, using surrogate **Random Forest** models trained on DFT data for fast evaluation.

## 🔍 Overview

The GA evolves a population of candidate solutions, using selection, crossover, and mutation to discover optimal structures with desired properties (e.g., volume, tetragonality, magnetic characteristics).

## 📂 File

- `optimizator_GA.py` – the main script for GA-based optimization

## ⚙️ How It Works

1. Initialize a population of candidate structures
2. Use surrogate models (from `trained_models/`) to evaluate fitness
3. Select the fittest candidates
4. Apply crossover and mutation to generate the next generation
5. Repeat for several iterations

## 🚀 Running the Optimizer

```bash
python optimizator_GA.py
```

Ensure that:
- The `trained_models/` folder contains `.pickle` models
- Input data (e.g., `df.csv`) is present if required by the script

## 📄 License

See the `LICENSE` file for details.


## 📁 Project Structure

```
dft-evolutionary-optimizer/
│
├── random_forest_train/          # Active learning and RF surrogate models
│   ├── active_learning_mag_aust.py
│   ├── active_learning_mag_mart.py
│   ├── active_learning_tetr.py
│   ├── active_learning_vol.py
│
├── sampler_for_dft/              # DFT input generation and sampling
│   ├── sampler_dft_random.py
│   ├── sampler_dft_uniform.py
│   ├── model_tetr.pickle
│   ├── model_volume.pickle
│   └── initial_poscars/
│       ├── POSCAR_216
│       └── POSCAR_225
│
├── trained_models/               # Pre-trained ML models
│   ├── model_mag_aust.pickle
│   ├── model_mag_mart.pickle
│   ├── model_tetr.pickle
│   └── model_volume.pickle
│
├── df.csv                        # Dataset
├── Curie.json                    # Dataset of Curie temperatures for all-d Heuslers
├── optimizator_GA.py             # Genetic Algorithm optimizer
└── LICENSE
```

## 📝 Citation

This code was used in the following publication:

> D. R. Baigutlin, et al.  
> "Machine learning algorithms for optimization of magnetocaloric effect in all-d-metal Heusler alloys."  
> *Journal of Applied Physics*, **136**(18), 2024.  
> [https://pubs.aip.org/aip/jap/article/136/18/183903/3319583](https://pubs.aip.org/aip/jap/article/136/18/183903/3319583)

If you use this code, please cite the article above.
