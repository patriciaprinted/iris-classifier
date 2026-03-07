from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


def main() -> None:
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)
    print("Training labels shape:", y_train.shape)
    print("Test labels shape:", y_test.shape)

    # Initialize + train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "outputs/model.joblib")


    # Predict + evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model accuracy:", accuracy)
    print("Predictions:", y_pred[:5])
    print("True labels:", y_test[:5])

    # Tree visualization
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True
    )
    plt.title("Decision Tree - Iris Classifier")
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()