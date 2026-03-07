# Iris Classifier (Decision Tree)

## Overview

This project demonstrates a simple machine learning workflow using the classic **Iris dataset** and a **Decision Tree classifier** built with **scikit-learn**.

The goal of the project is to classify iris flowers into three species based on their measurements:

* Setosa
* Versicolor
* Virginica

The model is trained, evaluated, and visualized using Python.

---

## Project Structure

```
iris-classifier/
│
├── src/
│   └── train.py        # Main training script
│
├── requirements.txt    # Python dependencies
├── .gitignore          # Files excluded from version control
└── README.md           # Project documentation
```

---

## Installation

Clone the repository:

```
git clone https://github.com/patriciaprinted/iris-classifier.git
cd iris-classifier
```

Create a virtual environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Run the Model

```
python src/train.py
```

The script will:

1. Load the Iris dataset
2. Split data into training and test sets
3. Train a Decision Tree classifier
4. Evaluate model accuracy
5. Visualize the decision tree

---

## Example Output

```
Training features shape: (120, 4)
Test features shape: (30, 4)
Model accuracy: 1.0
```

---

## Technologies Used

* Python
* scikit-learn
* matplotlib
* NumPy

---

## Future Improvements

Possible extensions include:

* Adding cross-validation
* Trying additional models (Random Forest, SVM)
* Hyperparameter tuning
* Model performance metrics

---

## License

MIT License
