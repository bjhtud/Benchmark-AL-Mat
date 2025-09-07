# Benchmark-AL-Mat: Active Learning Benchmark Framework for Materials Science

A comprehensive active learning benchmark framework specifically designed for materials science regression tasks. This project implements 20+ active learning strategies across multiple materials science datasets, providing a standardized evaluation platform for active learning research in materials discovery.

[//]: # (## 🌟 Key Features)

[//]: # ()
[//]: # (- **🔬 Domain-Specific**: Specialized optimization for materials science regression tasks)

[//]: # (- **📊 Rich Strategies**: 20+ active learning strategies including uncertainty sampling, query by committee, diversity methods, etc.)

[//]: # (- **🗄️ Diverse Datasets**: 14 materials science datasets covering concrete, steel, aluminum alloys, and more)

[//]: # (- **⚙️ Flexible Configuration**: Support for command-line parameters and JSON configuration files)

[//]: # (- **🔄 Reproducibility**: Complete random seed control and standardized evaluation protocols)

[//]: # (- **📈 Comprehensive Evaluation**: Detailed experiment tracking and time analysis)

[//]: # (- **🎯 Modular Design**: Easy to extend with new strategies and datasets)

## Supported Active Learning Strategies

### Baseline Methods
- **RandomSearch**: Random sampling baseline

### Gaussian Process-Based Methods
- **GaussianProcessBased**: GP-based uncertainty sampling
- **GSBAG**: Gaussian Process BAG method

### Tree-Based Methods
- **TreeBasedRegressor_Diversity**: Tree-based diversity sampling
- **TreeBasedRegressor_Representativity**: Tree-based representativeness sampling
- **TreeBasedRegressor_*_self**: Adaptive tree methods

### Committee-Based Methods
- **QueryByCommittee**: Committee disagreement query
- **RD_QBC_ALR**: Dimensionality reduction committee query

### Deep Learning Methods
- **LearningLoss**: Learning loss prediction
- **EGAL**: Expected Gradient Length
- **mcdropout**: Monte Carlo Dropout

### Bayesian Methods
- **BMDAL**: Bayesian Batch Mode Active Learning

### Geometric and Diversity Methods
- **GSi/GSx/GSy**: Geometric sampling variants
- **QDD**: Query Density Diversity

### Dimensionality Reduction Methods
- **Basic_RD_ALR**: Basic dimensionality reduction active learning
- **RD_GS_ALR**: Dimensionality reduction geometric sampling
- **RD_EMCM_ALR**: Dimensionality reduction expected model change maximization

## 📊 Dataset Overview

Currently, the project includes the uci-concrete dataset:

| Dataset | Description | Target Property |
|---------|-------------|-----------------|
| **Concrete Materials** | | |
| uci-concrete | UCI concrete strength data | Compressive strength |

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Benchmark-AL-Mat.git
cd Benchmark-AL-Mat

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Navigate to source code directory
cd src

# Run a single experiment (using random search strategy and uci-concrete dataset)
python main.py --random-state 42 --strategy RandomSearch --dataset uci-concrete

# Use different initialization method
python main.py --random-state 42 --strategy GSBAG --dataset uci-concrete --initial-method kmeans

# Custom query parameters
python main.py --random-state 42 --strategy RD_EMCM_ALR --dataset uci-concrete --n-pro-query 20
```

### Using Configuration Files

1. Create configuration file `config.json`:
```json
{
  "random_state": 42,
  "initial_method": "random",
  "strategy": "RD_EMCM_ALR",
  "dataset": "uci-concrete",
  "n_pro_query": 15,
  "threshold": 0.9
}
```

2. Run experiment:
```bash
python main.py --config-file config.json
```

## Detailed Parameter Description

### Command Line Arguments

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--random-state` | int | Random seed (required) | - |
| `--strategy` | str | Strategy name | - |
| `--dataset` | str | Dataset name | - |
| `--initial-method` | str | Initial sampling method | 'random' |
| `--n-pro-query` | int | Number of samples per query | 10 |
| `--threshold` | float | Early stopping threshold | 0.85 |
| `--config-file` | str | JSON configuration file path | None |

### Initialization Methods

- **random**: Random initialization
- **greedy_search**: Greedy search initialization
- **kmeans**: K-means clustering initialization
- **ncc**: Nearest centroid classifier initialization

## 🏗️ Project Structure

```
Benchmark-AL-Mat/
├── src/                         # Source code
│   ├── main.py                 # Main experiment script
│   ├── strategies/              # Active learning strategies
│   │   ├── __init__.py
│   │   ├── randomsearch.py     # Random search baseline
│   │   ├── gaussianprocess.py  # Gaussian process methods
│   │   ├── qbc.py              # Query by committee
│   │   ├── LL4AL.py            # Learning loss
│   │   ├── egal.py             # Expected gradient length
│   │   ├── bmdal.py            # Bayesian methods
│   │   └── ...                 # Other strategies
│   ├── utils/                   # Utility modules
│   │   ├── active_learner.py   # Active learning framework
│   │   ├── dataset_process.py  # Data processing
│   │   ├── initialize.py       # Initialization methods
│   │   └── utils_initialize/   # Initialization utilities
│   └── bmdal_reg/              # BMDAL implementation
├── dataset/                     # Datasets
│   ├── meta.csv                # Dataset metadata
│   └── uci-concrete/           # UCI concrete data
├── requirements.txt            # Dependencies
├── config_example.json         # Configuration example
└── README.md                   # This file
```

## Result Output

Experiment results are saved in the following structure:
```
result/
├── {n_pro_query}/              # Grouped by query count
│   ├── {random_state}/         # Grouped by random seed
│   │   └── {initial_method}/   # Grouped by initialization method
│   │       ├── {strategy}_{dataset}_{timestamp}.json
│   │       └── time_record/
│   │           └── {strategy}_{dataset}_{timestamp}.json
```

### Result File Contents
- **Main results**: R² score sequences, query indices, model performance changes
- **Time records**: Per-step query timing, overall runtime analysis

## About meta.csv File
The `meta.csv` file defines configuration information for datasets and should contain the following columns:

- `dataname`: Dataset name to reference in command line or config file
- `path`: Relative path to the dataset CSV file
- `target_columns`: Target columns to be excluded from features (semicolon-separated if multiple)
- `target_to_fit`: Target column(s) to be predicted (semicolon-separated if multiple)

Example (for uci-concrete dataset):
```csv
dataname,path,target_columns,target_to_fit
uci-concrete,../dataset/uci-concrete/concrete_data.csv,concrete_compressive_strength,concrete_compressive_strength
```

To add a new dataset, add a new row to `meta.csv` and ensure the corresponding data files are placed in the dataset directory.

## 🛠️ Extension Development

### Adding New Active Learning Strategies

1. Create new strategy file in `src/strategies/`
2. Implement strategy class with `query()` method
3. Import in `src/strategies/__init__.py`
4. Add to strategy registry in `main.py`

Example strategy template:
```python
class NewStrategy:
    def __init__(self, random_state=None):
        self.random_state = random_state
    
    def fit(self, X, y):
        # Train model
        pass
    
    def query(self, X_unlabeled, n_act=1):
        # Select most informative samples
        return selected_indices
```

### Adding New Datasets

1. Place data files in appropriate `dataset/` directory
2. Add dataset information to `dataset/meta.csv`
3. Ensure data format compatibility with existing datasets

## 🎯 Experimental Recommendations

### Benchmark Testing Workflow
1. **Baseline comparison**: First run RandomSearch as baseline
2. **Strategy evaluation**: Systematically test different strategy performances
3. **Parameter tuning**: Adjust n_pro_query and threshold
4. **Multiple runs**: Use different random seeds to ensure result reliability

### Performance Optimization Suggestions
- Small datasets: n_pro_query=5-10
- Large datasets: n_pro_query=20-50
- GPU acceleration: Deep learning strategies automatically use GPU (if available)

## 🔧 Dependencies

### Core Dependencies
- **numpy>=1.21.0**: Numerical computation
- **pandas>=1.3.0**: Data processing
- **scikit-learn>=1.0.0**: Machine learning algorithms
- **torch>=1.9.0**: Deep learning framework
- **scipy>=1.7.0**: Gaussian process kernels

### Optional Dependencies
- **matplotlib>=3.4.0**: Result visualization
- **seaborn>=0.11.0**: Statistical plots
- **xgboost**: XGBoost regressor support

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{benchmark-al-mat-2024,
  title={A Comprehensive Benchmark of Active Learning Strategies with AutoML for Small-Sample Regression in Materials Science},
  author={Jinghou Bi},
  email={jinghou.bi@tu-dresden.de},
  year={2025},
  url={https://github.com/bjhtud/Benchmark-AL-Mat}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Issue Reporting

Please report issues on the [GitHub Issues](https://github.com/bjhtud/Benchmark-AL-Mat) page.

## Contact

- **Author**: Jinghou Bi
- **Email**: jinghou.bi@tu-dresden.de
- **Institution**: TU Dresden

## Acknowledgments

Thanks to the materials science community for providing datasets, and to open source libraries like scikit-learn and PyTorch for their support.
