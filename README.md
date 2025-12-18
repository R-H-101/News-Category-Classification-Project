# Newsgroups Text Classification: Traditional ML vs. Neural Networks

## Details
- **Course:** Machine Learning 
- **Name:** Syed Raahem Haamer - 510552  
- **Date:** December 17, 2025  

---

## Abstract
This project compares the performance of traditional machine learning models (Support Vector Machine and Random Forest) with a dense neural network for multi-class text classification on the **20 Newsgroups dataset**. The goal is to categorize news articles into six topics: `comp.graphics`, `rec.autos`, `rec.motorcycles`, `sci.crypt`, `sci.electronics`, and `sci.space`. Key findings indicate that the neural network achieves the highest F1-score (0.8036), while SVM offers the best trade-off between speed and performance. The study includes detailed feature analysis, embedding, and practical recommendations for model selection.

---

## Introduction
### Problem Statement
Text classification is a fundamental task in natural language processing (NLP) with applications in content moderation, news categorization, and recommendation systems. The challenge lies in effectively extracting meaningful features from unstructured text and building models that generalize well across diverse categories.

### Objectives
1. Implement and compare traditional ML models (SVM, Random Forest) with a deep learning approach for text classification.
2. Evaluate the impact of feature engineering (TF-IDF, Chi-squared selection) on model performance.
3. Analyze learned word embeddings for semantic relationships.
4. Provide actionable insights on model selection based on accuracy, training time, and interpretability.

---

## Dataset Description
### Source
The **Newsgroups dataset** (sklearn.datasets.fetch_20newsgroups)

### Subset Used
- **Categories (6):** `comp.graphics`, `rec.autos`, `rec.motorcycles`, `sci.crypt`, `sci.electronics`, `sci.space`
- **Total Documents:** 5,921
- **Train/Test Split:** 80/20 (4,736 training, 1,185 testing)
- **Class Distribution:** Balanced (~790 documents per class)

### Preprocessing
1. **Text Cleaning:** Removed headers, footers, and quotes.
2. **Feature Extraction (Traditional ML):**
   - TF-IDF vectorization with 5,000 features (unigrams and bigrams).
     - TF-IDF stands for Term Frequency-Inverse Document Frequency. It's a statistical measure that evaluates how important a word is to a document in a collection or corpus. Words that appear frequently in a document but rarely in other documents get high scores. Common words ("the", "and", "is") get low scores because they appear everywhere. Also creates document embeddings.
     - TF = (Total number of terms in document d( / (Number of times term t appears in document d)
     - IDF = log(Number of documents containing term t / Total number of documents in corpus D)
     - TF-IDF(t,d,D)=TF(t,d) × IDF(t,D)
   - Chi-squared feature selection (top 1,000 features).
     - It's a statistical test that measures the dependence between a feature and the class label. Features that are highly dependent on the class are selected. If a word appears much more frequently in one category than others, it's probably important for classification.
     - Sum_of ((O-E)^2/E)
     - O = Observed frequency (actual counts: A, B, C, D)
     - E = Expected frequency (if feature and class were independent)
     - Feature Selection Process:
          - Compute χ² score for each term against each class.
          - For multi-class: Use one-vs-rest approach or aggregate scores.
          - Select top-k terms with highest χ² scores as features.
          - Use these selected features to train your model.
            
3. **Neural Network Preparation:**
   - Tokenization with a vocabulary size of 10,000.
   - Sequence padding/truncation to 500 tokens.
   - One-hot encoding of labels.

---

## Methodology

### Classical Machine Learning Approaches
1. **Support Vector Machine (SVM)**
   - Kernel: Linear (Transforms data to find a boundary between classes)
   - Hyperparameter Tuning (GridSearchCV) (Builds a grid of every possible combination of settings listed and tests them all to find the winner)
     - Regularization (`C`): [0.1, 1, 10, 100]. Higher C means more prone to overfitting and vice versa.
   - Cross-validation: Stratified 5-fold (Train on 4 and test on 1, repeating this 5 times).

2. **Random Forest**
   - Hyperparameter Tuning (GridSearchCV):
     - `n_estimators`: [100, 200] (Number of trees in the forest)
     - `max_depth`: [None, 50, 100]
     - `min_samples_split`: [2, 5, 10] (Minimum number of samples required to split an internal node)
     - `max_features`: ['sqrt', 'log2']
     - Class weight: [None, 'balanced']
   - Cross-validation: Stratified 5-fold (Train on 4 and test on 1, repeating this 5 times).

### Deep Learning Architecture
- **Model Type:** Dense Neural Network with Embedding
- **Architecture:**
  - Embedding Layer (10,000 vocab → 128-dim)
  - Global Average Pooling (for parameter reduction)
  - Dense Layers: 256 → 128 → 64 (ReLU, BatchNorm, Dropout)
  - Output Layer: 6 units (Softmax)
- **Regularization:** L2 regularization, Dropout (0.3–0.5)
- **Optimizer:** Adam (LR=0.001)
- **Callbacks:** Early Stopping, ReduceLROnPlateau (Reduce Learning Rate on Plateau).

### Hyperparameter Tuning Strategies
- **GridSearchCV** for SVM and Random Forest (exhaustive search over parameter grids for finding the optimal hyperparameters).
- **Stratified K-Fold Cross-Validation** (5 folds) to ensure representative validation.
- **Early Stopping** and **Learning Rate Reduction** for neural network training.

---

## Results & Analysis

### Performance Comparison
| Model             | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|-------------------|----------|-----------|--------|----------|-------------------|
| SVM               | 0.7848   | 0.8097    | 0.7848 | 0.7899   | 49.75             |
| Random Forest     | 0.7755   | 0.8013    | 0.7755 | 0.7810   | 681.07            |
| Neural Network    | 0.8025   | 0.8198    | 0.8025 | 0.8036   | 58.66             |

### Key Findings
- **Best Overall:** Neural Network (F1: 0.8036, Accuracy: 0.8025)
- **Fastest Training:** SVM (49.75 seconds)
- **Feature Importance (Random Forest):** Top terms include `car`, `bike`, `space`, `graphics`, `encryption`—aligning with the topic categories.

### Statistical Significance
- SVM and Neural Network outperform Random Forest in both accuracy and F1-score.
- Neural Network shows a **2.2% improvement** in F1 over SVM, though with slightly longer training time.

### Business Impact Analysis
- **SVM:** Recommended for rapid prototyping and high-dimensional text data.
- **Random Forest:** Useful when interpretability (feature importance) is critical.
- **Neural Network:** Best for production where accuracy is prioritized and computational resources are available.

### Visualization Highlights
1. **Accuracy/F1 Comparison:** Neural Network leads in both metrics.
2. **Training Time:** SVM is significantly faster than Random Forest; Neural Network is slightly slower than SVM.
3. **Embedding Analysis:** Words like `computer` are close to `graphics` and `3d`; `car` is associated with `auto` and `toyota`.
4. **Model Ranking:** Neural Network ranked first in combined score (accuracy 40% + F1 40% + speed 20%).

---

## Conclusion & Future Work
### Conclusion
- Neural networks provide the best classification performance for this text categorization task, albeit with increased training time compared to SVM.
- Traditional ML models remain competitive and may be preferred in resource-constrained or interpretability-focused scenarios.

### Future Work
1. **Advanced Architectures:** Can Experiment Transformers, or pre-trained embeddings (BERT, GPT).
3. **Hyperparameter Optimization:** Use Bayesian Optimization or Hyperband for more efficient tuning.
4. **Deployment Pipeline:** Build an end-to-end API for real-time news classification.

---

## References
1. 20 Newsgroups Dataset. http://qwone.com/~jason/20Newsgroups/
2. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly, 2019.
