import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_dir = "/Users/aurorekouakou/image_recognition/images"
csv_file = "/Users/aurorekouakou/image_recognition/styles.csv"
output_dir = "/Users/aurorekouakou/image_recognition/organized_dataset"

validation_split = 0.2  

for split in ['training', 'validation']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

df = pd.read_csv(csv_file, on_bad_lines='skip')

df['image_path'] = df['id'].astype(str) + ".jpg"
df = df[df['image_path'].apply(lambda x: os.path.exists(os.path.join(dataset_dir, x)))]
df_filtered = df.groupby('articleType').filter(lambda x: len(x) > 1)

article_types = df['articleType'].unique()
for split in ['training', 'validation']:
    for article in article_types:
        os.makedirs(os.path.join(output_dir, split, article), exist_ok=True)

train_data, val_data = train_test_split(df_filtered, test_size=validation_split, random_state=42, stratify=df_filtered['articleType'])

def organize_images(data, split):
    for _, row in data.iterrows():
        src = os.path.join(dataset_dir, row['image_path'])
        dest = os.path.join(output_dir, split, row['articleType'], row['image_path'])
        shutil.copy(src, dest)

print("Organized images for  training...")
organize_images(train_data, 'training')

print("Organized images for  validation...")
organize_images(val_data, 'validation')

print("Organization finished. images are ok at :", output_dir)
