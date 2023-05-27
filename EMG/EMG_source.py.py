import os
import numpy as np
from scipy import signal
from scipy.fft import fft,fftfreq
import matplotlib.pyplot as plt

def openEmg(file):
#Zugriff auf 2. Kanal: result[:,0]
#Zugriff auf 3. Kanal: result[:,1]

    try:
        meas=open(file,'rt')
        textlines=meas.readlines()
    except:
        print('Fehler beim Öffnen der Datei',file,"!")
        return np.array([])

    nSamples=len(textlines)
    nChannels=len(textlines[1].split())

    result=np.zeros((nSamples,nChannels))

    for r in range(0,nSamples):
        line=textlines[r]
        line=line.replace(',','.')
        words=line.split()

        if len(words)!=nChannels:
            print('Fehler beim Lesen von Zeile',r,'!')
            return result
        else:
            for c in range(0,nChannels):
                result[r,c]=float(words[c])

    return result

def plausibility(data):
##Prüfung auf Plausibilität (Amplitude des Signals im Bereich von 1uV-10mV)

    negbool=False
    posbool=False

    EMGmin=min(data)
    EMGmax=max(data)

    #Prüfung ob Messdaten innerhalb der Grenzen liegen
    if((EMGmin>=(-10000))and (EMGmin<=(-1))):
        negbool=True
    if((EMGmax<=10000)and(EMGmax>=1)):
        posbool=True

    if posbool:
        print('Positive Amplitude ist plausibel')
    else:
        print('Positive Amplitude ist nicht plausibel')

    if negbool:
        print('Negative Amplitude ist plausibel')
    else:
        print('Negative Amplitude ist nicht plausibel')

    return

def zeitinformation(data,fs):
##Generierung der Zeitinformation

    l=len(data)                            #Maximale Sampleanzahl
    t=np.arange(start=0,stop=l,step=1)     #1D-Matrix von 0 bis maximale Samplezahl
    t_f=np.zeros(l)                        #Leere 1D

    #Berechnung der Zeit für jede Samplenummer
    for x in t:
        t_f[x]=t[x]/fs

    return t_f

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def MVC(data1,data2,data3):
## Berechnung des MVC

    #Den absoluten Wert der Daten nehmen
    emg1=abs(data1)
    emg2=abs(data2)
    emg3=abs(data3)

    #Bestimmung des Maximums der einzelnen Messung
    max1=max(emg1)
    max2=max(emg2)
    max3=max(emg3)

    mvc=max(max1,max2,max3)
    print()
    print(f'MVC={mvc}')

    return mvc 

def spectral(data,fs):
    ##Bestimmung des Frequenzbereiches durch FFT

    #Maximale Sampleanzahl
    N1=len(data)

    #Fouriertransformation, arrayform
    signal=fft(data)
    
    #Fourietransformation, corresponding frequency
    freq=fftfreq(N1,1/fs)

    return signal,freq

def zeitparameter(data,mvc):
    ##Berechnung der Parameter im Zeitbereich

    #Berechnung des Mean Amplitude Value
    mav = np.mean(np.abs(data))
    print()
    print(f'mav={mav}')

    #Berechnung des Root Mean Square
    rms = np.sqrt(np.mean(data) ** 2)
    print()
    print(f'rms={rms}')

    #Berechnung des prozentualen MVC
    pMVC = max(data) / mvc * 100
    print()
    print(f'pMVC={pMVC}')

    return

def frequenzparameter(data):
    ##Berechnung der Parameter im Frequenzbereich

    #Berechnung der Mittlere Leistung
    P_mit = np.mean(np.abs(data))
    print()
    print(f'P_mit={P_mit}')

    #Berechnung der Medianfrequenz
    SMF_50 = np.sum(np.abs(data) )*0.5
    print()
    print(f'SMF_50={SMF_50}')

    #Berechnung der Eckfrequenz
    SEF_95 = np.sum(np.abs(data) )*0.95
    print()
    print(f'SEF_95={SEF_95}')

    return

def visualize(data,data_filtered,data_spectral,data_spectral_filtered,time,freq):
##Visualisierung der Messdaten
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    # font size of title
    plt.rcParams['axes.titlesize'] = 10
    axs[0, 0].plot(time, data)
    axs[0, 0].set_title('EMG-Signal im Zeitbereich (ungefiltert)')
    axs[0, 0].set_xlabel('Zeit in s')
    axs[0, 0].set_ylabel('Amplitude in \u03BCV')
    axs[0, 0].grid(True)
    axs[0, 0].set_xlim(0,1)
    
    axs[0, 1].plot(time, data_filtered)
    axs[0, 1].set_title('EMG-Signal im Zeitbereich (gefiltert)')
    axs[0, 1].set_xlabel('Zeit in s')
    axs[0, 1].set_ylabel('Amplitude in \u03BCV')
    axs[0, 1].grid(True)
    axs[0, 1].set_xlim(0,1)    
    
    axs[1, 0].plot(freq, np.abs(data_spectral))
    axs[1, 0].set_title('Fourietransformation des EMG_signals (ungefiltert)')
    axs[1, 0].set_xlabel('Frequenz in Hz')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].grid(True)
    axs[1, 0].set_xlim(0,1000)
    
    axs[1, 1].plot(freq, np.abs(data_spectral_filtered))
    axs[1, 1].set_title('Fourietransformation des EMG_signals (gefiltert)')
    axs[1, 1].set_xlabel('Frequenz in Hz')
    axs[1, 1].set_ylabel('Amplitude')
    axs[1, 1].grid(True)
    axs[1, 1].set_xlim(0,1000)
    plt.show()
    
    return

def relative_path(path):
    return os.path.join(os.path.dirname(__file__), path)
    
def main():
    
    # define data directory path
    data_path = '../Group2/'

    #Abtastfrequenz
    fs=2048
    lowcut=20
    highcut=500
    
    #Laden der Dateien für Berechnung des MVC
    emg_mvc1=openEmg(relative_path(f'{data_path}2.3.1'))
    emg_mvc2=openEmg(relative_path(f'{data_path}2.3.2'))
    emg_mvc3=openEmg(relative_path(f'{data_path}2.3.3'))

    #Berechnung des MVC
    mvc=MVC(emg_mvc1[:,1],emg_mvc2[:,1],emg_mvc3[:,1])
    #path for reading the data
    path = relative_path(f'{data_path}2.3.3')
    data=openEmg(path)

    #data[:,0]-linker Arm, data[:,1]-Rechter Arm
    data=data[:,1]

    plausibility(data)

    time=zeitinformation(data,fs)

    data_filtered=butter_bandpass_filter(data, lowcut, highcut, fs, order=4)

    data_spectral,freq=spectral(data,fs)
    data_spectral_filtered,freq=spectral(data_filtered,fs)
    print(f'--- Zeitparameter DATA ---')
    zeitparameter(data,mvc)
    print(f'--- Zeitparameter DATA_SPECTRAL ---')
    zeitparameter(data_spectral,mvc)

    print(f'--- Frequenzparameter DATA ---')
    frequenzparameter(data_spectral)
    print(f'--- Frequenzparameter DATA_SPECTRAL ---')
    frequenzparameter(data_spectral_filtered)

    visualize(data,data_filtered,data_spectral,data_spectral_filtered,time,freq)

if __name__=="__main__":
    main()