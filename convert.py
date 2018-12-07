from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pydub import AudioSegment

file = '/home/flash/Documents/IUBBooks/MLSP/Prooject/acoustic-analysis/project/training_dataset/angry/40.mp3'

if file.endswith(".mp3"):
	#print "training_dataset/angry/"+file
	sound=AudioSegment.from_mp3(file)
	sound.export("try.wav",format="wav")
