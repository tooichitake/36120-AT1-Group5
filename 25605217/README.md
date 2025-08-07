# NBAPredict - NBA Draft Prediction Package

[![PyPI - Version](https://img.shields.io/pypi/v/nbapredict)](https://test.pypi.org/project/nbapredict/)
[![Python Version](https://img.shields.io/badge/python-3.11.4-blue)](https://www.python.org/downloads/release/python-3114/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning package for predicting NBA draft prospects using college basketball statistics. Achieves >99% AUC using state-of-the-art gradient boosting models.

## 🚀 Installation

```bash
pip install -i https://test.pypi.org/simple/ nbapredict
```

For development:
```bash
git clone https://github.com/tooichitake/36120-AT1-Group5.git
cd 36120-AT1-Group5/25605217
pip install -r requirements.txt
```

## 📊 Quick Start

```python
import nbapredict as nbp

# Load your player data
player_data = pd.read_csv('college_players.csv')

# Get predictions using pre-trained models
predictions = nbp.predict_draft_probability(player_data)

# View results
print(predictions[['player_name', 'draft_probability']].head())
```

## 🎯 Features

- **Pre-trained Models**: LightGBM, CatBoost, and Random Forest models ready to use
- **High Accuracy**: >99% AUC on test data
- **Comprehensive Features**: 60+ engineered features from college statistics
- **Easy Integration**: Simple API for predictions
- **Jupyter Support**: Interactive notebooks for experimentation

## 📈 Model Performance

| Model | AUC | Precision | Recall | Training Time |
|-------|-----|-----------|--------|---------------|
| LightGBM | 0.9966 | 0.90 | 0.50 | 2 min |
| CatBoost | 0.9952 | 0.48 | 0.72 | 5 min |
| Random Forest | 0.9951 | 0.57 | 0.72 | 1 min |

## 📝 Usage Examples

### Basic Prediction

```python
import nbapredict as nbp
from nbapredict import NBAPrediction

# Initialize predictor
predictor = NBAPrediction(model='lightgbm')

# Predict single player
player_stats = {
    'GP': 35,
    'Min_per': 28.5,
    'TS_per': 0.585,
    'AST_per': 15.2,
    'height_inches': 78,
    'conf': 'ACC'
}
probability = predictor.predict_single(player_stats)
print(f"Draft probability: {probability:.2%}")
```

### Batch Prediction

```python
# Predict multiple players
predictions = predictor.predict_batch('players_2025.csv')
predictions.to_csv('draft_predictions.csv', index=False)
```

### Custom Threshold Analysis

```python
# Analyze at different confidence levels
thresholds = [0.3, 0.5, 0.7, 0.9]
analysis = predictor.analyze_thresholds(predictions, thresholds)
print(analysis)
```

## 🛠️ Advanced Features

### Feature Engineering

```python
from nbapredict.features import engineer_features

# Apply feature engineering
enhanced_data = engineer_features(raw_data)
```

### Model Training

```python
from nbapredict.modeling import train

# Train your own model
trainer = train.LightGBMTrainer(random_state=42)
model = trainer.train_with_optuna(X_train, y_train, X_val, y_val)
```

### Visualization

```python
from nbapredict.visualization import plot_predictions

# Visualize predictions
plot_predictions(predictions, save_path='predictions.png')
```

## 📦 Package Structure

```
nbapredict/
├── __init__.py          # Package initialization
├── dataset.py           # Data loading and preprocessing
├── features.py          # Feature engineering
├── visualization.py     # Plotting and analysis
└── modeling/           # Model training and prediction
    ├── __init__.py
    ├── train.py
    └── predict.py
```

## 🔧 Requirements

- Python 3.11.4 (exact version required)
- pandas >= 2.2.2
- scikit-learn >= 1.5.1
- lightgbm >= 4.4.0
- catboost >= 1.2.8
- numpy >= 1.26.0

## 📊 Data Format

Input data should include these features:
- **Basic Stats**: GP, Min_per, Ortg, usg, eFG, TS_per
- **Advanced Stats**: AST_per, TO_per, BPM, WS_40
- **Physical**: height (in format "6-5" for 6'5")
- **Context**: team, conf, yr (Fr/So/Jr/Sr)

## 🎓 Academic Context

Developed as part of UTS 36120 Data Science Practice (2025 Spring).

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Zhiyuan Zhao**  
Email: 25605217@student.uts.edu.au  
GitHub: [@tooichitake](https://github.com/tooichitake)

## 🙏 Acknowledgments

- UTS Faculty for guidance
- NBA and NCAA for inspiring this analysis
- Open source community for amazing ML libraries

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{nbapredict_2025,
  author = {Zhao, Zhiyuan},
  title = {NBAPredict: Machine Learning for NBA Draft Prospect Analysis},
  year = {2025},
  url = {https://github.com/tooichitake/36120-AT1-Group5}
}
```

## 🔗 Links

- [Documentation](https://github.com/tooichitake/36120-AT1-Group5/wiki)
- [Issue Tracker](https://github.com/tooichitake/36120-AT1-Group5/issues)
- [TestPyPI Package](https://test.pypi.org/project/nbapredict/)