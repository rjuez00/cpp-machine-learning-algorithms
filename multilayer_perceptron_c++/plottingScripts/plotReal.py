import matplotlib.pyplot as plt
import sys
import pandas

usage = """
USO:
    1.- ./exe outputimageName.jpg "FIGURE TITLE" "FIGURE SUBTITLE" input_file1 input_file2 input_file3...
    input_file sigue el formato de "filename.label.output"
"""

if len(sys.argv) < 4:
    print(usage)
    exit(1)


NUMERO_EPOCHS = 0

def read_input(filename):
    input = pandas.read_csv(filename, header = None, sep = " ", comment = "#", names = ["iteration", "ecm", "accTrain", "accVal"])
    with open(filename, "r") as file:
        accuracyTest = float(file.readlines()[-1].split("#")[1])
        input["accuracyTestFinal"] = accuracyTest
    
    return input.iloc[:, 1:]


fig, axs = plt.subplots(3,1)

for inputFile in sys.argv[4:]:
    df = read_input(inputFile)

    fileLabel = inputFile.split(".")[1]
    
    axs[0].plot(range(len(df["ecm"])), df["ecm"], label = fileLabel + f"; accuracyTest: {df['accuracyTestFinal'][0]}")
       

    axs[1].plot(range(len(df["accTrain"])), df["accTrain"], label = fileLabel + f"; accuracyTest: {df['accuracyTestFinal'][0]}")
    if df["accVal"][0] != -1:
        axs[2].plot(range(len(df["accVal"])), df["accVal"], label = fileLabel + f"; accuracyTest: {df['accuracyTestFinal'][0]}")

    NUMERO_EPOCHS = max(NUMERO_EPOCHS, len(df["accTrain"]))

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'center right')



axs[0].set_ylabel("ECM")
axs[0].set_xlabel("# Epoch")
axs[0].set_xlim(-2,NUMERO_EPOCHS)
#axs[0].legend()


axs[1].set_ylabel("Accuracy Train")
axs[1].set_xlabel("# Epoch")
#axs[1].legend()
axs[1].set_xlim(-2,NUMERO_EPOCHS)


axs[2].set_ylabel("Accuracy Validacion")
axs[2].set_xlabel("# Epoch")
#axs[2].legend()
axs[2].set_xlim(-2,NUMERO_EPOCHS)

fig.suptitle(sys.argv[2] + "\n" + sys.argv[3])

fig.set_size_inches(8,11.2)
plt.tight_layout()
plt.savefig(sys.argv[1])