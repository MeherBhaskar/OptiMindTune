# OptiMindTune: Your AI-Powered Hyperparameter Maestro 🧠✨

**Tired of manual hyperparameter tuning? Let intelligent AI agents do the heavy lifting!**

OptiMindTune is a proof-of-concept multi-agent system that leverages the power of Google's Gemini models to intelligently recommend and evaluate hyperparameter configurations for scikit-learn classification models. It's like having a team of AI data scientists collaborating to find the sweet spot for your machine learning models.

---

## 🚀 What's Inside?

OptiMindTune employs two specialized AI agents built using the Google ADK (Agent Development Kit):

1.  **🤖 Recommender Agent:**
    * Analyzes your dataset's metadata (number of samples, features, classes, class balance).
    * Considers previous evaluation results.
    * Intelligently suggests 1-2 scikit-learn classification models (RandomForest, LogisticRegression, SVC for now) and promising hyperparameter configurations to try next.
    * Provides reasoning for its recommendations, powered by Gemini.

2.  **🧐 Evaluator Agent:**
    * Takes the Recommender's suggestions.
    * Uses a dedicated `EvaluateModelTool` to train and evaluate the specified model with the given hyperparameters on your dataset using cross-validation (standard scaling is applied).
    * Returns the accuracy score.
    * Leverages Gemini to analyze the accuracy, dataset characteristics, and previous results to decide whether to "accept" the current model or suggest that the Recommender try different approaches.
    * Outputs its findings in a structured JSON format.

**The Loop:** These agents work in tandem. The Recommender suggests, the Evaluator tests, and based on the Evaluator's feedback (including accuracy and its LLM-generated reasoning), the Recommender makes new, more informed suggestions. This iterative process continues until a satisfactory model is found or a maximum number of iterations is reached.

---

## ✨ Key Features

* **LLM-Powered Intelligence:** Goes beyond simple grid search by using Gemini's reasoning capabilities for both recommendation and evaluation analysis.
* **Multi-Agent Collaboration:** Demonstrates how specialized agents can work together on a complex task.
* **Automated Workflow:** Reduces the manual effort and intuition traditionally required for hyperparameter tuning.
* **Extensible:** Built with the Google ADK, making it a good base for adding more models, evaluation metrics, or even more sophisticated agent interactions.
* **Clear Logging & Feedback:** Get insights into the decision-making process of each agent.
* **Focus on Scikit-learn:** Integrates directly with popular scikit-learn classifiers.

---


## ✨ Watch it in Action! ✨

[![Demo](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fyoutu.be%2F1LF6DrkTce4)](https://youtu.be/1LF6DrkTce4)

## 🛠️ Tech Stack

* **Python 3.10+**
* **Google Generative AI SDK (for Gemini)** (`google-generativeai`)
* **Google ADK (Agent Development Kit)** (`google-adk`) 
* **Scikit-learn** (`scikit-learn`)
* **Pandas** (`pandas`)
* **Dotenv** (`python-dotenv`) for environment variable management.

---
## ⚙️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://your-repository-url/OptiMindTune.git
    cd OptiMindTune
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory of the project:
    ```env
    GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
    # Add any other necessary API keys or configurations
    ```
    You can obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 🚀 Running OptiMindTune

Execute the main script to start the hyperparameter optimization process:

```bash
python main.py
```

The script will load the Iris dataset (by default) and initiate the Recommender and Evaluator agents. You'll see logs from each agent detailing:

- Recommendations made.
- Hyperparameters being evaluated.
- Cross-validation accuracy scores.
- The Evaluator agent's decision (accept/reject) and reasoning.
- The process will continue for a set number of iterations or until an accuracy threshold is met. Final results will be printed to the console.

## 💡 How It Works: The Optimization Loop
1. **Initialization**: `main.py` loads the dataset (e.g., Iris) and initializes `RecommenderAgent` and `EvaluatorAgent`.
2. **Recommendation**: The `RecommenderAgent` is called.
    - It receives dataset metadata and any previous results.
    - It queries Gemini to suggest 1-2 `(model_name, hyperparameters_string, reasoning) `sets.
3. **Parsing & Preparation**: main.py parses the Recommender's JSON response.
4. **Evaluation**: For each valid recommendation:
    - The `EvaluatorAgent` is invoked with the model_name, parsed hyperparameters_dict, and previous_results.
    - The `EvaluatorAgent` prompts Gemini, instructing it to use the EvaluateModelTool.
    - The `EvaluateModelTool` performs 5-fold cross-validation on the scaled data.
    - The accuracy is returned to the Gemini model within the EvaluatorAgent.
    - Gemini then provides a JSON response: {"accuracy": ..., "accept": ..., "reasoning": ...}.
5. **Decision & Iteration**:
    - `main.py `logs the evaluation result.
    - If a model is `accepted` and meets the `accuracy_threshold`, the optimization stops.
    - Otherwise, the results from this iteration become `previous_results` for the next call to the `RecommenderAgent`.
6. **Termination**: The loop stops if the `accuracy_threshold` is met or `max_iterations` are reached.

## 🔮 Future Enhancements & Roadmap

- [ ] **Support for More Models:** Expand beyond RandomForest, LogisticRegression, and SVC to include a wider array of scikit-learn classifiers (e.g., Gradient Boosting, XGBoost, LightGBM, Naive Bayes) and potentially models from other libraries.
- [ ] **Regression & Other Task Types:** Adapt the agent logic and evaluation tools to support regression tasks, and eventually other machine learning tasks like clustering or anomaly detection.
- [ ] **Advanced Evaluation Metrics:** Allow users to specify and optimize for different evaluation metrics beyond accuracy (e.g., F1-score (macro, micro, weighted), Precision, Recall, ROC AUC, LogLoss, Mean Squared Error for regression).
- [ ] **More Sophisticated Dataset Analysis:** Enhance the Recommender agent's ability to understand dataset nuances by:
    -   Performing more detailed automated exploratory data analysis (EDA).
    -   Considering feature types (categorical, numerical), presence of missing values, and feature scaling needs more explicitly in its recommendations.
    -   Potentially suggesting preprocessing steps.
- [ ] **Interactive Mode/UI:** Develop a simple web interface (e.g., using Streamlit or Flask) to:
    -   Allow users to upload their datasets.
    -   Configure optimization parameters.
    -   Monitor the optimization process in real-time.
    -   Visualize results and agent interactions.
- [ ] **Parallel Evaluation:** Implement the capability to evaluate multiple hyperparameter configurations recommended by the Recommender agent simultaneously, speeding up the optimization process.
- [ ] **Budget Constraints:** Allow users to define constraints for the optimization process, such as:
    -   Maximum runtime.
    -   Maximum number of model evaluations.
    -   Early stopping if no improvement is seen after a certain number of iterations.
- [ ] **Error Handling & Robustness:**
    -   Improve resilience to unexpected LLM outputs (e.g., malformed JSON, irrelevant suggestions).
    -   Add more robust error handling within the model evaluation tool for issues like convergence failures or incompatible hyperparameter combinations.
    -   Implement retry mechanisms for API calls.
- [ ] **Customizable Prompts & Agent Behavior:**
    -   Allow advanced users to tweak the system prompts for the Recommender and Evaluator agents to fine-tune their behavior.
    -   Offer different "strategies" for the Recommender (e.g., prioritize exploration vs. exploitation).
- [ ] **Result Persistence & Comparison:**
    -   Save optimization results to a file or a simple database.
    -   Allow users to load and compare results from different runs.
- [ ] **Integration with Experiment Tracking Tools:** Add support for logging results to popular experiment tracking platforms like MLflow or Weights & Biases.
- [ ] **Hyperparameter Space Definition:** Allow users to define or constrain the hyperparameter search space for each model type.
