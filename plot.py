import librosa
import librosa.display
import matplotlib.pyplot as plt

chunk1,sr = librosa.load('chunk-00.wav',sr = None)

chunk2,sr = librosa.load('chunk-01.wav',sr=None)

chunk3,sr = librosa.load('chunk-02.wav',sr = None)

chunk4,sr = librosa.load('chunk-06.wav',sr=None)

# plt.figure()

# plt.subplot(3, 1, 1)
# plt.figure(figsize=(15, 5))

file ,sr= librosa.load('test.wav',sr = None)
librosa.display.waveplot(file, sr=sr)
plt.title('File')
plt.show()


librosa.display.waveplot(chunk1, sr=sr)
plt.title('Chunk 1')
plt.show()

librosa.display.waveplot(chunk2, sr=sr)
plt.title('Chunk 2')
plt.show()

librosa.display.waveplot(chunk3, sr=sr)
plt.title('Chunk 3')
plt.show()

librosa.display.waveplot(chunk4, sr=sr)
plt.title('Chunk 4')
plt.show()