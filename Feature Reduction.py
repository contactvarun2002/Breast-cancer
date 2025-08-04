import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import random
# --------------------------
# Configuration
# --------------------------
input_csv = '/content/drive/MyDrive/Colab Notebooks/archive - 2025-08-01T210435.399/vit_features_with_labels.csv'     # ✅ Change this
output_csv = '/content/drive/MyDrive/Colab Notebooks/mms_selected_features.csv'   # ✅ Change this
label_column = 'label'                    # ✅ Set your label column name
num_features_to_select = 540               # ✅ Number of random features (excluding label)

# --------------------------
# Load CSV
# --------------------------
df = pd.read_csv(input_csv)

# --------------------------
# Ensure label column exists
# --------------------------
if label_column not in df.columns:
    raise ValueError(f"'{label_column}' column not found in CSV.")

# ---------------- Fitness Function ----------------
def fitness_function(selected_features):
    if np.sum(selected_features) == 0:
        return 0
    selected_idx = np.where(selected_features == 1)[0]
    X_sel = X[:, selected_idx]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# ---------------- Initialize Agents ----------------
def initialize_agents(num_agents, dim):
    return np.random.randint(0, 2, size=(num_agents, dim))

# ---------------- Update Mantis Position ----------------
def update_mantis_position(agent, best, iteration, max_iter):
    beta = 0.2
    rand = np.random.rand(agent.size)
    t = iteration / max_iter
    inertia = (1 - t) * agent
    attraction = t * best
    flip = np.where(rand < beta, 1 - agent, agent)
    return np.where(np.random.rand(agent.size) < 0.5, inertia, attraction).astype(int)

# ---------------- MMS Optimization ----------------
def MMS(num_agents, max_iterations, dim):
    agents = initialize_agents(num_agents, dim)
    best_agent = agents[0]
    best_fitness = fitness_function(best_agent)

    for iter in tqdm(range(max_iterations), desc="Running MMS"):
        for i in range(num_agents):
            fitness = fitness_function(agents[i])
            if fitness > best_fitness:
                best_fitness = fitness
                best_agent = agents[i].copy()
        for i in range(num_agents):
            agents[i] = update_mantis_position(agents[i], best_agent, iter, max_iterations)

    return best_agent
# Select feature columns
# --------------------------
feature_columns = [col for col in df.columns if col != label_column]
selected_features = random.sample(feature_columns, min(num_features_to_select, len(feature_columns)))

# --------------------------
# Final selected columns
# --------------------------
final_columns = selected_features + [label_column]
selected_df = df[final_columns]

# --------------------------
# Save to new CSV
# --------------------------
selected_df.to_csv(output_csv, index=False)
print(f"✅ Saved {len(selected_features)} random features + label to: {output_csv}")