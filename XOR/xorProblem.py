"""
XOR Problemi - Çok Katmanlı Yapay Sinir Ağı
============================================
Bu kod, XOR problemini gizli katmanlı bir MLP ile çözer.
Tek katmanlı perceptronun XOR'u öğrenemeyeceği gösterilir,
ardından iki katmanlı ağ ile problem başarıyla çözülür.
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
#  Aktivasyon fonksiyonları
# ─────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ─────────────────────────────────────────
#  XOR veri seti
# ─────────────────────────────────────────

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# ─────────────────────────────────────────
#  Çok Katmanlı Perceptron (2-2-1)
# ─────────────────────────────────────────

class MLP:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, lr=0.1):
        np.random.seed(42)
        self.lr = lr
        # Ağırlıklar ve bias'lar
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.loss_history = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        # Output katmanı gradyanları
        dz2 = (self.a2 - y) * sigmoid_derivative(self.z2)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        # Gizli katman gradyanları
        dz1 = (dz2 @ self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        # Güncelleme
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=20000):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            self.loss_history.append(loss)
            self.backward(X, y)
            if (epoch + 1) % 2000 == 0:
                print(f"Epoch {epoch+1:>6} | Loss: {loss:.6f}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# ─────────────────────────────────────────
#  Tek Katmanlı Perceptron (karşılaştırma)
# ─────────────────────────────────────────

class SingleLayerPerceptron:
    def __init__(self, input_size=2, lr=0.1):
        np.random.seed(42)
        self.lr = lr
        self.W = np.random.randn(input_size, 1)
        self.b = np.zeros((1, 1))
        self.loss_history = []

    def forward(self, X):
        self.z = X @ self.W + self.b
        self.a = sigmoid(self.z)
        return self.a

    def backward(self, X, y):
        m = X.shape[0]
        dz = (self.a - y) * sigmoid_derivative(self.z)
        self.W -= self.lr * (X.T @ dz / m)
        self.b -= self.lr * (np.sum(dz, axis=0, keepdims=True) / m)

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            self.loss_history.append(loss)
            self.backward(X, y)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# ─────────────────────────────────────────
#  Eğitim ve Sonuçlar
# ─────────────────────────────────────────

print("=" * 50)
print("TEK KATMANLI PERCEPTRON (XOR öğrenemez)")
print("=" * 50)
slp = SingleLayerPerceptron(lr=0.5)
slp.train(X, y, epochs=20000)
slp_preds = slp.predict(X)
print(f"\nTahminler : {slp_preds.flatten()}")
print(f"Gerçek    : {y.flatten()}")
slp_acc = np.mean(slp_preds == y) * 100
print(f"Doğruluk  : %{slp_acc:.1f}")

print("\n" + "=" * 50)
print("ÇOK KATMANLI PERCEPTRON (XOR öğrenir)")
print("=" * 50)
mlp = MLP(hidden_size=4, lr=1.0)
mlp.train(X, y, epochs=20000)
mlp_preds = mlp.predict(X)
print(f"\nTahminler : {mlp_preds.flatten()}")
print(f"Gerçek    : {y.flatten()}")
mlp_acc = np.mean(mlp_preds == y) * 100
print(f"Doğruluk  : %{mlp_acc:.1f}")


# ─────────────────────────────────────────
#  Görselleştirme
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("XOR Problemi – Sinir Ağı Analizi", fontsize=14, fontweight="bold")

# 1) Kayıp eğrileri
axes[0].plot(slp.loss_history, label="Tek Katman", color="red", alpha=0.7)
axes[0].plot(mlp.loss_history, label="Çok Katman (MLP)", color="blue", alpha=0.7)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Kayıp Eğrisi Karşılaştırması")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2) MLP karar sınırı
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                     np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = mlp.forward(grid).reshape(xx.shape)
axes[1].contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.7)
axes[1].contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
colors = ["green" if label == 0 else "orange" for label in y.flatten()]
axes[1].scatter(X[:, 0], X[:, 1], c=colors, s=150, zorder=5, edgecolors="k")
for i, (xi, yi) in enumerate(X):
    axes[1].annotate(f"({xi},{yi})→{y[i,0]}", (xi + 0.05, yi + 0.05), fontsize=9)
axes[1].set_title("MLP Karar Sınırı")
axes[1].set_xlabel("x₁")
axes[1].set_ylabel("x₂")

# 3) Doğruluk karşılaştırması
models = ["Tek Katman\n(Perceptron)", "Çok Katman\n(MLP)"]
accs   = [slp_acc, mlp_acc]
bar_colors = ["#e74c3c", "#2ecc71"]
bars = axes[2].bar(models, accs, color=bar_colors, width=0.4, edgecolor="black")
axes[2].set_ylim(0, 110)
axes[2].set_ylabel("Doğruluk (%)")
axes[2].set_title("Model Doğruluk Karşılaştırması")
for bar, acc in zip(bars, accs):
    axes[2].text(bar.get_x() + bar.get_width() / 2, acc + 2,
                 f"%{acc:.0f}", ha="center", fontweight="bold", fontsize=12)
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("xor_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGrafik 'xor_results.png' olarak kaydedildi.")