import math
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import numpy as np
import pandas as pd
import librosa
import librosa.display
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks
from scipy.stats import linregress
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def multifrac(audio_path):

    y, sr = librosa.load(audio_path, sr=None)

    def moving_average(y, s, theta):
        N = len(y)
        y_moving_avg = np.zeros(N)
        past = int(np.floor((s - 1) * theta))
        future = int(np.ceil((s - 1) * (1 - theta)))
    
        for t in range(N):
            start = max(0, t - past)
            end = min(N, t + future + 1)
            y_moving_avg[t] = np.mean(y[start:end])
    
        return y_moving_avg

    def detrend(y, y_moving_average):
        return y - y_moving_average


    def rms(epsilon, s):
   
        N = len(epsilon)
        Ns = math.floor(N/s - 1)
   
        Fv_s = []
        for vi in range(Ns):
            v = vi+1
            l = (v-1)*s

            ini = l
            fin = s+l
            ev = epsilon[ini:fin]
            r_m_s = rms = np.sqrt(np.mean(ev ** 2))
            Fv_s.append(r_m_s)    
        return Fv_s

    def calculate_fluctuation(Fv, q, s):

        Fv = np.array(Fv)
    
        Ns = max(0, math.floor((N / s) - 1)) 
        if(q != 0):
            return (np.mean(Fv**q))**(1/q)
        else:
            return np.exp(np.mean(np.log(Fv)))




    def create_matrix(q_list, s_list, y):

        Fs = [] # F para cada s, dentro tem os F(q)
        for s in s_list:

            y_moving_average = moving_average(y, s, theta)
            epsilon = detrend(y, y_moving_average)
            Fvs = rms(epsilon, s)

            aux = []
            for q in q_list:
                Fvs = rms(epsilon, s)
                Fluc = calculate_fluctuation(Fvs, q, s)
                aux.append(Fluc)

            Fs.append(aux)

        line = len(s_list)
        colu = len(q_list)
        A = np.zeros([line, colu])
        for i in range(line):
            aux = Fs[i]
            for j in range(colu):
                A[i,j] = aux[j]
        return A

    N = len(y)
    theta = 1

    q_list = np.linspace(-5,5,11)
    s_list = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    A = create_matrix(q_list, s_list, y)

    hq = []
    log_s = np.log(s_list)
    for i in range(len(q_list)):
        F = A[:,i]
        log_F = np.log(F)
        slope, intercept, r_value, p_value, std_err = linregress(log_s, log_F)
        hq.append(slope)


    tau = hq*q_list-1
    alpha = np.gradient(tau,q_list)
    f_alp = q_list*alpha - tau


    delta_alfa = alpha[0]-alpha[-1]
    return delta_alfa

def extract_voice_features(audio_path):
    # Carregar áudio
    y, sr = librosa.load(audio_path, sr=None)

    # --- MFCCs (Mel-Frequency Cepstral Coefficients) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    chrome = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    zero_cros = np.array([np.mean(librosa.feature.zero_crossing_rate(y))])

    peaks, _ = find_peaks(y, height=0)  # height define um limiar para considerar um pico
    density = len(peaks)/len(y)
    density = np.array([density])

    delta_a = multifrac(audio_path)
    delta_a = np.array([delta_a])

    features = np.concatenate([mfccs_mean, chrome, contrast, zero_cros, density, delta_a])

    return features




df = pd.read_csv("../../dataset/torgodataset_Organized/torgodataset_organized.csv")
path_list = df['path']
label = df['label']

new_label = []
feature_list = []

N = len(path_list)
#N = 1000
base_path = "../../dataset/torgodataset_Organized/"
for i in range(N):
    aux = path_list[i]
    pht = base_path + aux[37:]
    
    try:
        # Tentar carregar o arquivo de áudio
        y, sr = librosa.load(pht, sr=None)
        features = extract_voice_features(pht)
        feature_list.append(features)
        new_label.append(label[i])
        


    except Exception as e:
        # Caso ocorra um erro, imprimir uma mensagem e continuar
        print(f"Erro ao carregar o arquivo {pht}: {e}")
        continue

    if(i%500==0):
        print(f"{round(100*(i)/N,2)}%")
print(f"{round(100*(i+1)/N,2)}%")


X_train = feature_list
y_train = new_label

# Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


torch.save(X_train_tensor, 'X_train_tensor.pth')
torch.save(y_train_tensor, 'y_train_tensor.pth')
torch.save(X_test_tensor, 'X_test_tensor.pth')
torch.save(y_test_tensor, 'y_test_tensor.pth')



# Carregar os tensores de volta
X_train_tensor = torch.load('X_train_tensor.pth')
y_train_tensor = torch.load('y_train_tensor.pth')
X_test_tensor = torch.load('X_test_tensor.pth')
y_test_tensor = torch.load('y_test_tensor.pth')


# Aplicar SMOTE apenas no conjunto de treinamento
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tensor.numpy(), y_train_tensor.numpy())

# Convertendo para tensores novamente
X_train_tensor_resampled = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor_resampled = torch.tensor(y_train_resampled, dtype=torch.float32)

# O conjunto de teste permanece inalterado
X_test_tensor = torch.tensor(X_test_tensor, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_tensor, dtype=torch.float32)



X_train_tensor = X_train_tensor_resampled
y_train_tensor = y_train_tensor_resampled





# Definir o modelo (rede neural simples)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(35, 64)  # Camada de entrada (33 features) para camada escondida
        self.layer2 = nn.Linear(64, 64)  # Camada escondida 1 para camada escondida 2
        self.layer3 = nn.Linear(64, 1)   # Camada escondida 2 para saída binária
        self.sigmoid = nn.Sigmoid()      # Função de ativação para saída binária

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # Função ReLU na camada escondida 1
        x = torch.relu(self.layer2(x))  # Função ReLU na camada escondida 2
        x = self.sigmoid(self.layer3(x))  # Saída binária
        return x

model = SimpleNN()


los_list = []

# Definir a função de perda e o otimizador
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
epochs = 5000
for epoch in range(epochs):
    model.train()  # Colocar o modelo em modo de treinamento
    
    optimizer.zero_grad()  # Zerando os gradientes do otimizador
    
    # Passo para frente
    outputs = model(X_train_tensor).squeeze()  # Remover dimensões desnecessárias
    
    # Calcular a perda
    loss = criterion(outputs, y_train_tensor)
    
    # Passo para trás (backpropagation)
    loss.backward()
    
    # Atualizar os pesos
    optimizer.step()
    
    # Imprimir a perda a cada 10 épocas
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        los_list.append(loss.item())



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
plt.savefig('confusion.png')
plt.show()



# Matriz de confusão em porcentagem
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalização para porcentagem

plt.figure(figsize=(6, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
            xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusão (Porcentagem)')
plt.savefig('confusion_percent.png')
plt.show()




#Checando o balanceamento

label = y_train_tensor.numpy()

cont0 = 0
cont1 = 1
for i in range(len(label)):
    if(label[i] > 0.5):
        cont0+=1
    if(label[i] < 0.5):
        cont1+=1

print(cont0)
print(cont1)



x_epoch = np.linspace(0,epochs,len(los_list))
plt.plot(x_epoch, los_list)
plt.xlabel("Epoches")
plt.ylabel("Loss")
plt.savefig('loss.png')
plt.show()




# Salvar o modelo completo (estrutura + pesos)
torch.save(model, 'model_complete.pth')
