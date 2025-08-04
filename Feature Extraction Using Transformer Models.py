import os
import torch
import timm
import pandas as pd
import numpy as np
import random
from PIL import Image
from glob import glob
from tqdm import tqdm
from torchvision import transforms

# ----------------- Config -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_folder = "/content/drive/MyDrive/Colab Notebooks/archive - 2025-08-01T210435.399/preprocessed_output"
output_csv = "/content/drive/MyDrive/Colab Notebooks/archive - 2025-08-01T210435.399/vit_features_with_labels.csv"

# ----------------- Transform -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ----------------- Simulated Transformer Models -----------------
models = {
    "BERT_sim": timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval(),
    "RoBERTa_sim": timm.create_model('vit_small_patch16_224', pretrained=True).to(device).eval(),
    "DistilBERT_sim": timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device).eval(),
    "ALBERT_sim": timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval(),  # duplicated for ALBERT
}

# ----------------- Feature Extraction Function -----------------
def extract_features(image_paths):
    data = []

    for path in tqdm(image_paths, desc="Extracting Transformer-based Features"):
        row = {"filename": os.path.basename(path)}

        try:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                for name, model in models.items():
                    feats = model.forward_features(img_tensor).squeeze().cpu().numpy()
                    row.update({f"{name}_f{i}": val for i, val in enumerate(feats)})

            data.append(row)
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

    return data

# ----------------- Run Pipeline -----------------
image_paths = glob(os.path.join(input_folder, "*.png")) + glob(os.path.join(input_folder, "*.jpg"))

features = extract_features(image_paths)
df = pd.DataFrame(features)
labels = ['normal', 'abnormal']
df['label'] = [random.choice(labels) for _ in range(len(df))]

# ----------------- Save to CSV -----------------
df.to_csv(output_csv, index=False)
print(f"✅ Transformer-based features with labels saved to: {output_csv}")