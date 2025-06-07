# Naive Bayes & K-Nearest Neighbours (KNN) from Scratch – Titanic Dataset

This project implements **Naive Bayes** and **K-Nearest Neighbours (KNN)** classifiers **from scratch using only NumPy and Pandas**, applied to the Titanic dataset to predict passenger survival.

📁 File: `Titanic_NB_KNN_FromScratch.ipynb`

---

## 📌 Objectives

- Build classification models (Naive Bayes and KNN) **without using any ML libraries**.
- Preprocess and visualize the Titanic dataset.
- Evaluate and compare model performance.
- Gain a deeper understanding of how these algorithms work internally.

---

## 🧠 Algorithms

### ✅ Naive Bayes
- Based on Bayes’ Theorem with a Gaussian (normal) assumption for continuous features.
- Assumes features are **independent**.
- Calculates probabilities for each class and selects the most probable.

### ✅ K-Nearest Neighbours (KNN)
- A non-parametric, distance-based algorithm.
- For a given test sample, finds the `k` closest training examples using **Euclidean distance** and performs majority voting.

---

## 📊 Dataset

The [Titanic dataset](https://www.kaggle.com/competitions/titanic/data) contains information about passengers and whether they survived the Titanic disaster.

Features used:
- `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

Target:
- `Survived` (1 = survived, 0 = did not survive)

---

## 🧹 Preprocessing Steps

- Removed unused columns (`Name`, `Cabin`, `Ticket`, etc.)
- Filled missing `Age` and `Fare` with median values.
- Filled missing `Embarked` with the most frequent value.
- Encoded `Sex` and `Embarked` into numeric values.
- Split dataset into **training (70%)** and **testing (30%)** using `train_test_split`.

---

## ⚙️ Requirements

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn` (only used for train/test split and accuracy calculation)

Install using pip if needed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## ▶️ How to Run

1. Clone this repository or download the `.ipynb` file.
2. Open `Titanic_NB_KNN_FromScratch.ipynb` in Jupyter Notebook.
3. Run all cells from top to bottom.

---

## ✅ Results

- **Naive Bayes Accuracy**: ~0.79  
- **KNN Accuracy (k=5)**: ~0.81  
- (May vary slightly based on split and randomness.)

---

## 📌 Notes

- This implementation is for educational purposes to demonstrate how these algorithms work internally.
- No external ML libraries like `sklearn.naive_bayes` or `sklearn.neighbors` were used for modeling.

---

## 📎 Author

Japan N. Pancholi
