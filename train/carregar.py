import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Defina a classe do modelo
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Exemplo de camada
        self.fc2 = nn.Linear(50, 1)  # Exemplo de camada

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Ativação sigmoide para a saída binária
        return x

# Carregar os tensores do conjunto de teste
X_test_tensor = torch.load('X_test_tensor.pth')
y_test_tensor = torch.load('y_test_tensor.pth')

# Criar DataLoader para o conjunto de teste
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Carregar o modelo completo
model = torch.load('model_complete.pth', map_location='cpu')

# Testar o modelo
model.eval()  # Colocar o modelo em modo de avaliação
with torch.no_grad():  # Desligar o cálculo de gradientes durante a avaliação
    predicted = model(X_test_tensor).squeeze()  # Previsões no conjunto de teste
    predicted_class = (predicted > 0.5).float()  # Limite de 0.5 para previsão binária
    accuracy = (predicted_class == y_test_tensor).float().mean()  # Comparar com os rótulos de teste
    print(f'Accuracy on Test Set: {accuracy:.4f}')


y_test_numpy = y_test_tensor.numpy()
predicted_class_numpy = predicted_class.numpy()
cm = confusion_matrix(y_test_numpy, predicted_class_numpy)


# Plotar a matriz de confusão
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusão')
plt.show()



# Matriz de confusão em porcentagem
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalização para porcentagem

plt.figure(figsize=(6, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
            xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusão (Porcentagem)')
plt.show()
