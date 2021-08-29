import matplotlib.pyplot as plt


def graph_data(file_name, id):
    data = []
    file = open(file_name, "r", encoding='UTF-8')
    while True:
        text = file.readline()
        if not text:
            break
        data.append(float(text.split()[0]))

    print(data[-1])
    plt.plot(data, label=id)


plt.figure(1)
plt.grid()
graph_data("protected data/loss.txt", "loss")
plt.title("loss")
plt.legend(loc=0)
plt.xlabel("epoch")
plt.savefig("loss.jpeg")
plt.figure(2)
plt.grid()
graph_data("protected data/valid_f1_measure.txt", "vf1-measure")
plt.title("f1_measure")
plt.legend(loc=0)
plt.xlabel("epoch")
plt.savefig("f1_measure.jpeg")
plt.show()
plt.close(1)
plt.close(2)
