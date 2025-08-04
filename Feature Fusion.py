import numpy as np
import pandas as pd
# ----------------------------------------
# 🧠 AZO Optimization for Fusion
# ----------------------------------------
# ----------------------------
# Step 1: Load the CSVs
# ----------------------------
Dataset1_path = "/content/drive/MyDrive/Colab Notebooks/mms_selected_features.csv"
Dataset2_path = "/content/drive/MyDrive/Colab Notebooks/archive (97)/mms_selected_features.csv"

Dataset1 = pd.read_csv(Dataset1_path)
Dataset2 = pd.read_csv(Dataset2_path)

# ----------------------------
# Step 2: Align rows
# ----------------------------
min_len = min(len(Dataset1), len(Dataset2))
Dataset1 = Dataset1.iloc[:min_len]
Dataset2 = Dataset2.iloc[:min_len]

# ----------------------------
# Step 3: Concatenate
# ----------------------------
fused_df = pd.concat([Dataset1, Dataset2], axis=1)

# ----------------------------
# Step 4: Replace NaNs with 0
# ----------------------------
fused_df.fillna(0, inplace=True)

# ----------------------------
# Step 5: Labeling
# ----------------------------
normal_count = 500
total_rows = fused_df.shape[0]
labels = [0 if i < normal_count else 1 for i in range(total_rows)]
fused_df["label"] = labels
class AZO:
    def __init__(self, obj_func, num_features, pop_size=20, max_iter=50):
        self.obj_func = obj_func
        self.num_features = num_features
        self.pop_size = pop_size
        self.max_iter = max_iter

    def optimize(self):
        population = np.random.rand(self.pop_size, self.num_features)
        fitness = np.apply_along_axis(self.obj_func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1 = np.random.rand(self.num_features)
                r2 = np.random.rand(self.num_features)
                new_solution = population[i] + r1 * (best_solution - population[i]) + r2 * (np.random.rand(self.num_features) - 0.5)
                new_solution = np.clip(new_solution, 0, 1)
                new_fitness = self.obj_func(new_solution)

                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness
# ----------------------------
# Step 6: Save
# ----------------------------
output_path = "/content/drive/MyDrive/Colab Notebooks/archive (97)/fused_dataset1_dataset2.csv"
fused_df.to_csv(output_path, index=False)

# ----------------------------
# Step 7: Summary
# ----------------------------
print(f"✅ Fused Dataset1 + Dataset2 saved at: {output_path}")
print("🔢 Final shape:", fused_df.shape)
print(fused_df.head())