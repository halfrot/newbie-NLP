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


graph_data("vf1_measure.txt", "vf1-measure")
graph_data("tf1_measure.txt", "tf1-measure")
plt.legend(loc=0)
plt.xlabel("epoch")
plt.title("validation_set and test_set f1-measure")
plt.savefig("f1-measure.jpeg")
plt.show()
