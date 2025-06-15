import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential #type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore
from src.utils import load_data


def train_models():
    images, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    os.makedirs("models", exist_ok=True)

    # SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    print("SVM Accuracy:", accuracy_score(y_test, svm_model.predict(X_test)))
    pickle.dump(svm_model, open("models/svm_model.pkl", "wb"))

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    print("KNN Accuracy:", accuracy_score(y_test, knn_model.predict(X_test)))
    pickle.dump(knn_model, open("models/knn_model.pkl", "wb"))

    # Naïve Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    print("Naïve Bayes Accuracy:", accuracy_score(y_test, nb_model.predict(X_test)))
    pickle.dump(nb_model, open("models/nb_model.pkl", "wb"))

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    print("Decision Tree Accuracy:", accuracy_score(y_test, dt_model.predict(X_test)))
    pickle.dump(dt_model, open("models/dt_model.pkl", "wb"))

    # CNN
    X_train_cnn = X_train.reshape(-1, 64, 64, 1)
    X_test_cnn = X_test.reshape(-1, 64, 64, 1)
    y_train_cnn = to_categorical(y_train, num_classes=2)
    y_test_cnn = to_categorical(y_test, num_classes=2)

    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train_cnn, epochs=15, batch_size=32, validation_data=(X_test_cnn, y_test_cnn))

    _, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn)
    print("CNN Accuracy:", cnn_accuracy)
    cnn_model.save("models/cnn_model.h5")

if __name__ == "__main__":
    train_models()
