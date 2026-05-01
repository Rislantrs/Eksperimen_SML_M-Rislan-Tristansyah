import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

def load_data(file_path):
    print(f"Loading data: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df, target_col='depression_label'):
    print("Mulai preprocessing data...")
    df_clean = df.copy()
    
    # hapus duplikat
    df_clean = df_clean.drop_duplicates()
    
    # Pisahkan target jika ada
    target_series = None
    if target_col in df_clean.columns:
        target_series = df_clean[target_col].copy()
        df_clean = df_clean.drop(columns=[target_col])
    
    # Tangani kolom boolean secara eksplisit
    bool_cols = df_clean.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_clean[col] = df_clean[col].astype(int)

    # pisah kolom numerik dan teks (setelah drop target dan konversi bool)
    num_cols = df_clean.select_dtypes(include=['int64', 'float64', 'int32']).columns
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    
    # handle missing values
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df_clean[num_cols] = imputer_num.fit_transform(df_clean[num_cols])
        
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_clean[cat_cols] = imputer_cat.fit_transform(df_clean[cat_cols])
        
    # label encoding untuk kategorikal
    encoder = LabelEncoder()
    for col in cat_cols:
        df_clean[col] = encoder.fit_transform(df_clean[col])
        
    # scaling data numerik (Hanya fitur, bukan target)
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
    
    # Kembalikan target ke dataframe
    if target_series is not None:
        # Pastikan target adalah integer 0/1 (jika boolean atau float)
        df_clean[target_col] = target_series.astype(int).values
        
    return df_clean

if __name__ == "__main__":
    print("Mulai proses otomatis...")
    
    # setup path folder
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(base_dir) 
    
    input_file = os.path.join(root_dir, 'Teen_Mental_Health_Dataset.csv')
    output_file = os.path.join(base_dir, 'Teen_Mental_Health_Dataset_ready.csv')
    
    if os.path.exists(input_file):
        data = load_data(input_file)
        data_bersih = preprocess_data(data)
        
        # simpan hasil
        data_bersih.to_csv(output_file, index=False)
        print("Preprocessing selesai!")
        print(f"File tersimpan di: {output_file}")
    else:
        print(f"Error: file tidak ketemu di {input_file}")
