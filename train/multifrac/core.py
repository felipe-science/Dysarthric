import math
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress




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


df = pd.read_csv("../../dataset/torgodataset_Organized/torgodataset_organized.csv")
path_list = df['path']
label = df['label']



for i in range(10):

    aux = path_list[i]
    base_path = "../../dataset/torgodataset_Organized/"
    pht = base_path + aux[37:]

    da = multifrac(pht)
    print(da)