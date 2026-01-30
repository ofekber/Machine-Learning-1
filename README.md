# Machine Learning – Course Assignments

Repository for Machine Learning course homework (HW1–HW5). Each folder contains assignment materials, instructions, and code or notebooks where applicable.

---

## Project structure

### HW1
- **HW1.pdf** – Assignment questions
- **Instructions HW1.pdf** – Submission and formatting instructions

*PDF-only; no code.*

---

### HW2
- **HW2 dry.pdf** – Assignment (dry / written part)
- **Instuctions HW2.pdf** – Instructions (note: typo in filename)
- **HW wet.py** – Working solution: multiclass Perceptron classifier (NumPy, pandas)
- **hw2_files/**
  - **HW2_ID1_ID2.py** – Student template (replace IDs, implement `fit`/`predict`)
  - **iris_sep.csv** – Iris-style dataset (no header; last column = labels)

**Run (example):**
```bash
cd HW2
python "HW wet.py" hw2_files/iris_sep.csv
```

---

### HW3
- **HW3.pdf** – Assignment
- **Instruction HW3.pdf** – Instructions

*PDF-only; no code.*

---

### HW4
- **HW4.pdf** – Assignment
- **Instructions HW4.pdf** – Instructions
- **HW4_Q3.ipynb** – Jupyter notebook (Fashion-MNIST, sklearn, SVM, plotting)

**Requirements:** `numpy`, `pandas`, `matplotlib`, `scikit-learn` (and Jupyter).

---

### HW5
- **HW5.pdf** – Assignment
- **Instructions HW5.pdf** – Instructions
- **HW5.ipynb** – Jupyter notebook (wine quality, Decision Trees, Random Forest)

**Data:** Uses `winequality-red.csv` (not in repo; obtain per assignment instructions).

**Requirements:** `numpy`, `pandas`, `matplotlib`, `scikit-learn` (and Jupyter).

---

## Running code

- **HW2:** Python 3 with `numpy`, `pandas`. Run from `HW2` with CSV path as argument.
- **HW4, HW5:** Open the `.ipynb` in Jupyter and run cells; install dependencies above if needed.

---

## Notes

- Replace placeholder student IDs in `hw2_files/HW2_ID1_ID2.py` before submission.
- HW4 downloads Fashion-MNIST via `sklearn.datasets.fetch_openml`.
- HW5 expects `winequality-red.csv` in the notebook’s working directory (or adjust the path in the notebook).
