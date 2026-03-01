import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. VERİ YÜKLEME VE ÖN İŞLEME
df = pd.read_csv('data.csv')
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# M=1 (Malignant=kötü), B=0 (Benign=iyi)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# 2. VERİ BÖLME VE ÖLÇEKLENDİRME
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression için ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. LOGISTIC REGRESSION MODELİ
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Tahmin Yapma
y_pred_lr = lr_model.predict(X_test_scaled)

# 4. METRİKLERİ YAZDIRMA
print("--- LOGISTIC REGRESSION SONUÇLARI ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred_lr):.4f}")

# 5. KARMAŞIKLIK MATRİSİ GÖRSELLEŞTİRME
cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Benign (İyi)', 'Malignant (Kötü)'], 
            yticklabels=['Benign (İyi)', 'Malignant (Kötü)'])

plt.title('Logistic Regression - Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.tight_layout()

# Resmi Kaydet
plt.savefig('confusion_matrix_lr.png')
print("\nKarmaşıklık matrisi 'confusion_matrix_lr.png' oluşturuldu.")