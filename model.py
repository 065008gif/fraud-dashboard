import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


def load_and_preprocess_data(sample_size=20000):
    data = pd.read_csv("creditcard.csv")

    # Sample for fast training
    data = data.sample(n=sample_size, random_state=42)

    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def build_ann(input_dim, num_layers, neurons, learning_rate):
    model = Sequential()

    model.add(Dense(neurons, activation="relu", input_dim=input_dim))

    for _ in range(num_layers - 1):
        model.add(Dense(neurons, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_and_evaluate(num_layers, neurons, learning_rate, epochs):
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    model = build_ann(
        input_dim=X_train.shape[1],
        num_layers=num_layers,
        neurons=neurons,
        learning_rate=learning_rate
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )

    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype("int32")

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, recall, precision, f1, history, y_test, y_pred
