Project Proposal
Goal
Quantitatively compare multiple generations of transformer-based language models (GPT-2, GPT-3, GPT-4, and
GPT-5 if available) on their ability to detect cross-cultural communication norm violations. The project includes a
deep-learning component: training a lightweight neural probe (MLP) on model embeddings to evaluate
representational quality and improve detection performance.
Dataset
Use the labeled scenario dataset from Weng et al. (UMAP ’25) — 512 scenarios with gold labels (violation / no
violation), varying sender/receiver nationality, age, and gender. The dataset is preformatted for classification and
suitable for both zero-shot evaluation and training.
Approach
1. Zero-Shot Evaluation
Query GPT models (GPT-2 locally; GPT-3/4/5 via OpenAI API) with:
“Does this interaction contain a cross-cultural communication norm violation? Respond ‘Yes’ or ‘No.’”
Map responses to binary labels and compare accuracy across models.
2. Deep-Learning Component
Extract embeddings using a small transformer (e.g., distilbert-base-uncased or gpt2-small) on MacBook M2 Air.
Train a small MLP classifier on these embeddings to predict violations (train/val/test split: 70/15/15).
Evaluate performance using accuracy, precision, recall, specificity, and F1.
3. Statistical Analysis
Use chi-square and ANOVA/Kruskal-Wallis tests to compare model performance across demographic pairings.
Report effect sizes and p-values.
Measures of Success
Reproduce GPT-4 baseline trends from Weng et al.
Detect statistically significant performance differences between models.
Successfully train and evaluate the MLP probe.
Demonstrate insights into representation quality or improvements over baseline classifiers.
Resources
Hardware: MacBook M2 Air (sufficient for MLP training and small transformer inference).
Software: Python, PyTorch, Hugging Face Transformers, scikit-learn, NumPy, Matplotlib.
API Access: OpenAI API (funded by research lab).
Dataset: Weng et al. (UMAP ’25) — already uploaded.
Ethics: No human subjects; IRB exempt.
