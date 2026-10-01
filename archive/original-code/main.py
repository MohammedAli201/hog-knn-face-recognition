
import cv2
import os
import csv
from datetime import datetime
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time as Time


def load_training_images(training_dir):
    labels = []
    train_images = []

    for subdir in os.listdir(training_dir):
        subdir_path = os.path.join(training_dir, subdir)
        # print(subdir_path)
        if not os.path.isdir(subdir_path):
            continue
        paths = [os.path.join(subdir_path, f)
                 for f in os.listdir(os.path.join(subdir_path))]

        for i in range(len(paths)):

            image = cv2.imread(paths[i])
            train_images.append(image)
            dir_path = os.path.dirname(paths[i])
            folder_name = os.path.basename(dir_path)

            # Use a regular expression to extract the folder number
            extract_label = re.search(r'\d+', folder_name).group()
            labels.append(extract_label)
            # print(extract_label)
    train_images = np.array(train_images)
    label = [int(x) for x in labels]
    y = np.array(label)
    # Before return we should split the data into train and test

    return train_images, y


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        # Sort the faces based on their area (width * height)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        # Keep only the largest face (first face in the sorted list)
        faces = [faces[0]]

    return faces


def extract_features(image, faces):
    features = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        # print(f'Before resize {face.shape}')
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (64, 128),
                               interpolation=cv2.INTER_AREA)
       # print(f'After resize {face_gray.shape}')
        # Calculate gradients using Sobel operators
        Gx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate magnitude and angle
        magnitude = cv2.magnitude(Gx, Gy)
        angle = cv2.phase(Gx, Gy, angleInDegrees=True)

        # Compute HOG features using magnitude and angle
        hog_features = hog(magnitude, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(3, 3), block_norm='L2', feature_vector=True)

        features.append(hog_features)
    return features


def train_face_recognizer(training_dir):
    # Load training images
    train_images, y = load_training_images(training_dir)
    y = np.array(y)

    # Extract features from faces in training images
    features = []
    for image in train_images:
        faces = detect_faces(image)
        features.extend(extract_features(image, faces))
    X = np.array(features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42)
    # print("The shape of X_train :", X_train.shape)
    # Scale features
    # scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    # Train classifier
    classifier = KNeighborsClassifier(n_neighbors=5, algorithm='brute')

    # print("The shape of X_train :", X_train.shape)
    # print("The shape of y_train :", y_train.shape)
    classifier.fit(X_train, y_train)
    # Compute distance between training sample
    distances, indices = classifier.kneighbors(X_train)
    distance_threshold = np.mean(distances[:, -1])
    print("The distance threshold is :", distance_threshold)

    return classifier, y_train, X_test, y_test, distance_threshold, indices


def recognize_faces(image, classifier):

    faces = detect_faces(image)
    # print(faces)
    if len(faces) == 0:
        return [], []
    features = extract_features(image, faces)

    # features = extract_features(image, faces)

    X = np.array(features)
    # print(X.shape)
    X_scaled = scaler.transform(X)

    predicted_labels = classifier.predict(X_scaled)
    test_accuracy = classifier.score(
        scaler.transform(X), predicted_labels)
    print("The test accuracy is :", test_accuracy)
    print("The predict :", predicted_labels)
    return predicted_labels, faces


def predict_face(image, classifier, distance_threshold):

    # Detect faces in image
    faces = detect_faces(image)
    if len(faces) == 0:
        return [], []

    # Extract features from faces
    features = extract_features(image, faces)
    X = np.array(features)

    # Scale features
    X = scaler.transform(X)

    # Make predictions
    y_pred = classifier.predict(X)
    y_pred = y_pred.astype(object)

    # Compute distances between new face and its k nearest neighbors
    distances, _ = classifier.kneighbors(X)
    print("The distance is :", distances)
    # distances[:, -1] = 150
    # Set label to "unknown" if distance is above threshold
    distance_threshold += 10
    y_pred[distances[:, -1] > 140] = "unknown"

    return y_pred, faces


if __name__ == '__main__':
    webcam = True
    names_ = ["James", "John", "Robert", "Michael", "William", "David", "Joseph", "Charles",
              "Thomas", "Daniel", "Matthew", "Andrew", "Anthony", "Kenneth", "Steven", "Brian"]

    filename = 'attendance.csv'
    root_dir = 'data\Final Training Images'
    scaler = StandardScaler()
    classifier, y, X_test, y_test, distance_threshold, indices = train_face_recognizer(
        root_dir)
    y_pred = classifier.predict(scaler.transform(X_test))
    cm = confusion_matrix(y_test, y_pred)
    # visualize y_pred and y_test using plt.hist
    plt.hist(y_pred, bins=range(17), alpha=0.5, label='y_pred')
    plt.hist(y_test, bins=range(17), alpha=0.5, label='y_test')
    plt.legend(loc='upper right')
    plt.show()

    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar=False, annot_kws={'size': 20})
    plt.xticks(np.arange(16)+0.5, rotation=90, fontsize=15)
    plt.yticks(np.arange(16)+0.5, rotation=0, fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.title('Confusion matrix', fontsize=20)
    plt.show()

    # prediction, faces = recognize_faces(frame, classifier, scaler)
    test_accuracy = classifier.score(
        scaler.transform(X_test), y_test)
    print("The test accuracy is :", test_accuracy)
    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        if frame is None:
            print("Error reading frame from webcam")
            break

        # prediction, faces = recognize_faces(frame, classifier)
        pred, faces = predict_face(frame, classifier, distance_threshold)
        if len(faces) == 0:
            continue
        prediction_value = str(pred[0])

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if prediction_value == "unknown":
                cv2.putText(frame, prediction_value, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif prediction_value != "unknown":
                cv2.putText(frame, names_[int(prediction_value)], (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                is_duplicate = False
                with open(filename, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        print(row)
                        print(names_[int(prediction_value)])
                        print(prediction_value)
                        if row[0] == names_[int(prediction_value)] and row[1] == datetime.now().strftime('%Y-%m-%d'):
                            is_duplicate = True
                            print("Duplicate entry found")
                            break

                # Write to the CSV file only if it's not a duplicate entry
                if not is_duplicate:
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([names_[int(prediction_value)], datetime.now().strftime('%Y-%m-%d'),
                                        datetime.now().strftime('%H:%M:%S'), 'Present'])

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
