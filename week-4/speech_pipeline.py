
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr)
    y = y / np.max(np.abs(y))
    return y, sr

def plot_waveform(y, sr):
    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.show()

def extract_mel(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel)
    return mel_db

def extract_mfcc(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

class KeywordCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32*10*10, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
