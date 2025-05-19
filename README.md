# OptiMindTune: Multi-Agent AutoML with Gemini 🧠✨

A collaborative multi-agent system for intelligent hyperparameter optimization powered by Google's Gemini models.

## 🎯 Overview

OptiMindTune uses three specialized AI agents to automate the model selection and hyperparameter tuning process for scikit-learn classifiers. Each agent focuses on a specific aspect of the optimization process, creating a robust and intelligent AutoML system.

## 🤖 Agent Architecture

### 1. Recommender Agent
- Analyzes dataset characteristics
- Reviews past performance history
- Suggests models and hyperparameters
- Provides reasoning for recommendations
- Adapts to feedback from other agents

### 2. Evaluator Agent
- Handles model training and validation
- Implements cross-validation pipeline
- Manages data preprocessing
- Reports performance metrics
- Maintains model state

### 3. Decision Agent
- Evaluates model performance
- Makes accept/reject decisions
- Balances exploration/exploitation
- Guides optimization strategy
- Determines search termination

## 🔄 Optimization Loop

1. **Initialization**
   - Load and analyze dataset
   - Configure optimization parameters
   - Initialize agent communication

2. **Core Loop**
   - Recommender suggests model configurations
   - Evaluator tests suggestions
   - Decision agent guides next steps
   - Real-time logging of all interactions

3. **Termination**
   - Target accuracy achieved
   - Maximum iterations reached
   - Exploration ratio satisfied

## ⚙️ Features

- **Real-Time Logging:** Detailed agent interaction history
- **Configurable Goals:** Adjustable accuracy and exploration targets
- **Cross-Validation:** Robust model evaluation
- **Error Handling:** Graceful failure recovery
- **Conversation Tracking:** Complete interaction history

## 🛠️ Technical Stack

- Python 3.10+
- Google ADK & Gemini API
- scikit-learn
- pandas
- Async I/O

## 🚀 Quick Start

1. **Setup**
```bash
git clone https://github.com/MeherBhaskar/OptiMindTune.git
cd OptiMindTune
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Configuration**
```bash
# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env
```

3. **Run**
```bash
python main.py
```

## 📊 Output Structure

```
output/
└── conversations/                   # Agent interaction logs
    └── conversation_YYYYMMDD_HHMMSS.json
```

### Conversation Format
```json
{
  "timestamp": "ISO-8601 timestamp",
  "metadata": {
    "config": {
      "max_iterations": 5,
      "min_accuracy": 0.85,
      "target_accuracy": 0.95,
      "exploration_ratio": 0.3
    },
    "total_iterations": "actual iterations",
    "best_model": {
      "model": "model name",
      "hyperparameters": "param settings",
      "accuracy": "best score"
    }
  },
  "interactions": [
    {
      "timestamp": "ISO-8601 timestamp",
      "iteration": "iteration number",
      "agent": "agent name",
      "input": "agent input",
      "output": "agent response",
      "status": "success/failed"
    }
  ]
}
```

## 🔮 Future Enhancements

- Expanded model support (XGBoost, LightGBM)
- Regression task support
- Custom metric optimization
- Parallel evaluation
- Web interface
- MLflow/W&B integration
- Custom agent strategies
