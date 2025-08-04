import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------- Step 1: Configuration --------------------
csv_path = "/content/drive/MyDrive/Colab Notebooks/archive (97)/fused_dataset1_dataset2.csv"
label_column = "label"
test_size = 0.3
random_state = 42

# -------------------- Step 2: Read and Preprocess CSV --------------------
def parse_feature_string(s):
    try:
        # Remove square brackets if any, split by space, convert to float
        return np.array([float(val) for val in str(s).replace('[', '').replace(']', '').split()], dtype=np.float32)
    except:
        return np.array([], dtype=np.float32)

df = pd.read_csv(csv_path)

# Determine feature column(s)
feature_cols = [col for col in df.columns if col != label_column]

# Parse and combine features
parsed_features = df[feature_cols].applymap(parse_feature_string)
features = np.stack(parsed_features.apply(lambda row: np.concatenate(row.values), axis=1))

# Labels
labels = df[label_column].values
if not np.issubdtype(labels.dtype, np.number):
    labels = LabelEncoder().fit_transform(labels)

# Reshape for Conv1D: (samples, timesteps, channels)
features = features.reshape(features.shape[0], features.shape[1], 1)

# -------------------- Step 3: Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

# -------------------- Step 4: Lightweight CNN --------------------
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),

    Conv1D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -------------------- Step 5: Train Model --------------------
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# -------------------- Step 6: Evaluate --------------------
y_pred = np.argmax(model.predict(X_test), axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# -------------------- Step 7: Print --------------------
print("\nEvaluation Metrics:")
print(f"Accuracy  : {accuracy * 100:.3f}")
print(f"Precision : {precision * 100:.3f}")
print(f"Recall    : {recall * 100:.3f}")
print(f"F-measure : {f1 * 100:.3f}")