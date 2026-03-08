import numpy as np

# XOR Veri Seti
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Hiperparametreler
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

# Ağırlıkları rastgele başlatma
W1 = np.random.uniform(size=(input_size, hidden_size))
W2 = np.random.uniform(size=(hidden_size, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

print("Eğitim başlıyor...")

for i in range(epochs):
    # İleri Besleme (Forward Propagation)
    hidden_layer_input = np.dot(X, W1)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, W2)
    predicted_output = sigmoid(output_layer_input)
    
    # Geri Yayılım (Backpropagation)
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Ağırlık Güncelleme
    W2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    W1 += X.T.dot(d_hidden_layer) * learning_rate

print("\n--- Test Sonuçları ---")
for i in range(len(X)):
    print(f"Giriş: {X[i]} -> Tahmin: {predicted_output[i].round()} (Ham: {predicted_output[i]})")