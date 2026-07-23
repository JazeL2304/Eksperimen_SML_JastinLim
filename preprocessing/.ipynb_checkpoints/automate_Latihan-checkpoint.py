"""
automate_Latihan.py
Script otomatisasi preprocessing data COVID-19 Indonesia
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def kategori_risiko(new_cases):
    if new_cases < 10:
        return "Rendah"
    elif new_cases < 100:
        return "Sedang"
    else:
        return "Tinggi"


def preprocess_data(input_path, output_path):
    # 1. Load data
    df = pd.read_csv(input_path)

    # 2. Bersihkan kolom persen jadi angka desimal
    for kolom in ["Case Fatality Rate", "Case Recovered Rate"]:
        df[kolom] = df[kolom].astype(str).str.replace("%", "", regex=False)
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce") / 100

    # 3. Bikin kolom target
    df["Kategori_Risiko"] = df["New Cases"].apply(kategori_risiko)

    # 4. Pilih fitur
    fitur_dipakai = [
        "New Deaths", "New Recovered", "New Active Cases",
        "Population Density", "Population", "Area (km2)",
        "Case Fatality Rate", "Case Recovered Rate",
        "Kategori_Risiko"
    ]
    df_model = df[fitur_dipakai].copy()
    df_model = df_model.dropna()

    # 5. Encoding target
    le = LabelEncoder()
    df_model["Kategori_Risiko_Encoded"] = le.fit_transform(df_model["Kategori_Risiko"])

    # 6. Split fitur & target
    X = df_model.drop(columns=["Kategori_Risiko", "Kategori_Risiko_Encoded"])
    y = df_model["Kategori_Risiko_Encoded"]

    # 7. Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 9. Simpan hasil
    df_final = pd.DataFrame(X_train_scaled, columns=X.columns)
    df_final["target"] = y_train.values
    df_final.to_csv(output_path, index=False)

    print(f"Preprocessing selesai. Jumlah baris hasil: {len(df_final)}")
    return df_final


if __name__ == "__main__":
    preprocess_data(
        input_path="../namadataset_raw/covid_19_indonesia_time_series_all.csv",
        output_path="namadataset_preprocessing.csv"
    )