import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

df = pd.read_csv("dataset/vertebrate.csv")

## Preprocessing
lb = LabelBinarizer()
le = LabelEncoder()

# Body Temperature
df["Body Temperature"] = lb.fit_transform(df["Body Temperature"])
# Skin Cover
df["Skin Cover"] = le.fit_transform(df["Skin Cover"])
# Gives Birth 
df["Gives Birth"] = lb.fit_transform(df["Gives Birth"])
# Aquatic Creature
df["Aquatic Creature"] = le.fit_transform(df["Aquatic Creature"])
# Aerial Creature
df["Aerial Creature"] = lb.fit_transform(df["Aerial Creature"])
# Has Legs 
df["Has Legs"] = lb.fit_transform(df["Has Legs"])
# Hibernates 
df["Hibernates"] = lb.fit_transform(df["Hibernates"])

## Pemilahan Data untuk melakukan Peramalan
X = df.drop(["Name", "Class Label"],axis=1).iloc[:-1]
y = df["Class Label"].iloc[:-1]
# Data yang dilakukan untuk peramalaan
cases = df[df["Class Label"].isnull() == True].drop(["Name","Class Label"],axis=1)

# Training dan Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Model -> Menggunakan SVM
model = SVC()
model.fit(X_train, y_train)

# Melakukan Peramalan terhadap data yang bersangkutan
hasil_prediksi = model.predict(cases)
print(f"Hasil Prediksi dari yang bersangkutan : {hasil_prediksi}")

# Melakukan Peramalan terhadap data testing
testing_prediksi = model.predict(X_test)
print(f"Hasil Prediksi terhadap data X_test : {testing_prediksi} ")

# Menghitung akurasi data / persentase data
akurasi = accuracy_score(y_test, testing_prediksi)
print(f"Hasil Akurasi Data terhadap data testing : {akurasi*100:.2f}%")
