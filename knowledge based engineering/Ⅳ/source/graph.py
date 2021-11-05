import matplotlib.pyplot as plt


def graph_data(file_name, id):
    cnt = 0
    data = []
    file = open(file_name, "r", encoding='UTF-8')
    while True:
        cnt += 1
        text = file.readline()
        if not text:
            break
        data.append(float(text.split()[0]))

    print(data[-1])
    print(cnt)
    plt.plot(data, label=id)


plt.figure(1)
plt.grid()
graph_data("data3.3/loss.txt", "loss")
plt.title("loss")
plt.legend(loc=0)
plt.xlabel("epoch")
plt.savefig("loss.jpeg")
plt.figure(2)
plt.grid()
graph_data("data3.3/valid_f1.txt", "vf1-measure")
plt.title("f1_measure")
plt.legend(loc=0)
plt.xlabel("epoch")
plt.savefig("f1_measure.jpeg")
plt.show()
plt.close(1)
plt.close(2)
