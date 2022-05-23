import matplotlib.pyplot as plt
import sys
import pandas

"""
USO:
    1.- ./exe outputimageName.jpg "FIGURE TITLE" input_file1 input_file2 input_file3...
    input_file sigue el formato de "filename.label.output"
"""

NUMERO_EPOCHS = 0

def read_input(filename):
    input = pandas.read_csv(filename, header = None, sep = " ", comment = "#", names = ["iteration", "ecm", "accTrain", "accTest"])
    return input.iloc[:, 1:]

fig, axs = plt.subplots(3,1)

for inputFile in sys.argv[3:]:
    df = read_input(inputFile)

    fileLabel = inputFile.split(".")[1]
    
    axs[0].plot(range(len(df["ecm"])), df["ecm"], label = fileLabel)
   

    axs[1].plot(range(len(df["accTrain"])), df["accTrain"], label = fileLabel)
    if df["accTest"][0] != -1:
        axs[2].plot(range(len(df["accTest"])), df["accTest"], label = fileLabel)

    NUMERO_EPOCHS = max(NUMERO_EPOCHS, len(df["accTrain"]))


axs[0].set_ylabel("ECM")
axs[0].set_xlabel("# Epoch")
axs[0].set_xlim(-2,NUMERO_EPOCHS)
axs[0].legend()


axs[1].set_ylabel("Accuracy Train")
axs[1].set_xlabel("# Epoch")
axs[1].legend()
axs[1].set_xlim(-2,NUMERO_EPOCHS)


axs[2].set_ylabel("Accuracy Test")
axs[2].set_xlabel("# Epoch")
axs[2].legend()
axs[2].set_xlim(-2,NUMERO_EPOCHS)

fig.suptitle(sys.argv[2])

fig.set_size_inches(8,11.2)
plt.tight_layout()
plt.savefig(sys.argv[1])