import numpy as np
import predictors

def argsort(seq):
  return sorted(range(len(seq)), key=seq.__getitem__)

classes = ["White","Red","Orange","Yellow","Green","Cyan","Blue","Pink"]

while True:
  note = input("Enter a note (C to B, capital leters, add '#' or 'b' for sharps and flats): ")
  note_num = 0
  if note == "C":
    note_num = 1
  elif note == "C#" or note == "Db":
    note_num = 2
  elif note == "D":
    note_num = 3
  elif note == "D#" or note == "Eb":
    note_num = 4
  elif note == "E":
    note_num = 5
  elif note == "F":
    note_num = 6
  elif note == "F#" or note == "Gb":
    note_num = 7
  elif note == "G":
    note_num = 8
  elif note == "G#" or note == "Ab":
    note_num = 9
  elif note == "A":
    note_num = 10
  elif note == "A#" or note == "Bb":
    note_num = 11
  elif note == "B":
    note_num = 12
  else:
    note = input("Invalid note. Please try again: ")

  octave = int(input("Enter an octave number(1-7): "))
  print()

  frequencies = [32.70320, 34.64783, 36.70810, 38.89087, 41.20344, 43.65353, 46.24930, 48.99943, 51.91309, 55.00000, 58.27047, 61.73541]
  frequency = round(frequencies[note_num - 1]*(2**(octave-1)), 2)

  probs=[] # list of probabilities of the note being in the different color classes

  x = np.array([note_num,octave,frequency]).reshape(1,3)
  probs.append(predictors.predictWhite(x))
  probs.append(predictors.predictRed(x))
  probs.append(predictors.predictOrange(x))
  probs.append(predictors.predictYellow(x))
  probs.append(predictors.predictGreen(x))
  probs.append(predictors.predictCyan(x))
  probs.append(predictors.predictBlue(x))
  probs.append(predictors.predictPink(x))
  
  prediction = classes[np.argmax(probs)]
  print("PREDICTION -",prediction.upper())
  print()
  print("Breakdown:")
  for i in argsort(probs)[::-1]:
    print(classes[i]+":",str(round(probs[i]*100,2))+"%")
  print()
  input("Press return to try another note ")
