import numpy as np
import cv2
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Initialize the hand detector and classifier
detector = HandDetector(maxHands=2)
classifier = Classifier("keras_model.h5", "labels.txt")

# Define the categories
categories = {0: 'Hurts', 1: 'Iloveyou', 2: 'Pain', 3: 'Stop'}

def predict_sign_language():
    # Initialize video capture from the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)
    # Lists to store true labels and predictions for analysis
    true_labels = []
    predictions = []
    try:
        while True:
            _, frame = cap.read()
            hands, frame = detector.findHands(frame, False)
            if hands:
                # Get the prediction
                prediction_probs, predicted_index = classifier.getPrediction(frame, draw=True)
                predicted_class = categories[predicted_index]
                # Append the true label and prediction (simulating true labels here)
                true_label = categories[predicted_index]  # Replace this with actual true labels in a real scenario
                true_labels.append(true_label)
                predictions.append(predicted_class)
                # Print for debugging
                print(f"Prediction: {predicted_class}, True Label: {true_label}")
            cv2.imshow("Frame", frame)
            interrupt = cv2.waitKey(10)
            if interrupt & 0xFF == 27:  # esc key
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    print("True labels:", true_labels)
    print("Predictions:", predictions)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Model accuracy: {(accuracy * 100):.2f}%")
    
    return true_labels, predictions

# Plot accuracy
def plot_accuracy(accuracy):
    plt.figure(figsize=(5, 5))
    plt.bar(['Accuracy'], [accuracy * 100])
    plt.ylabel('Percentage')
    plt.title('Model Accuracy')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(true_labels, predictions, categories):
    cm = confusion_matrix(true_labels, predictions, labels=list(categories.values()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(categories.values()), yticklabels=list(categories.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Start prediction and get labels and predictions
true_labels, predictions = predict_sign_language()

# Visualize results
plot_accuracy(accuracy_score(true_labels, predictions))
plot_confusion_matrix(true_labels, predictions, categories)
