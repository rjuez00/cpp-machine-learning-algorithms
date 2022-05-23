import matplotlib.pyplot as plt
import sys
import pandas

NUMERO_EPOCHS = 150
output_folder = "images"

def read_input(filename):
    input = pandas.read_csv(filename, header = None, sep = " ", comment = "#", names = ["iteration", "ecm", "accTrain", "accTest"])
    return input.iloc[:, 1:]


fileName = sys.argv[1]
last_line = open(fileName).readlines()[-1]

df = read_input(fileName)

fig, axs = plt.subplots(2,1)
axs[0].plot(range(len(df["ecm"])), df["ecm"], label = last_line)
axs[0].set_ylabel("ECM")
axs[0].set_xlabel("# Epoch")
axs[0].set_xlim(-2,min(NUMERO_EPOCHS, len(df["ecm"])))

axs[1].plot(range(len(df["accTrain"])), df["accTrain"], label = "train accuracy")
if df["accTest"][0] != -1:
    axs[1].plot(range(len(df["accTest"])), df["accTest"], label = "test accuracy")
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("# Epoch")
axs[1].legend()
axs[1].set_xlim(-2,min(NUMERO_EPOCHS, len(df["ecm"])))


fig.suptitle(fileName.split(".")[1] + "\n" + last_line.split("#")[1])


plt.tight_layout()
plt.savefig(output_folder + "/" + fileName.split(".")[0] + ".jpg")