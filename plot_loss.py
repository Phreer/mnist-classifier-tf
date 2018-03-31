import csv
from matplotlib import pyplot as plt
import numpy as np

def parse(train_data):
    train_data = np.array(train_data, dtype=np.float32)
    train_data = train_data[:, 1:]
    return train_data.T

if __name__ == "__main__":

    with open(r"E:\Chrome Downloads\run_train1-tag-loss_loss0.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        items = list(reader)
        items.pop(0)
        train0 = parse(items[0:len(items)//3])
        train1 = parse(items[len(items)//3: len(items)//3*2])
        train2 = parse(items[len(items)//3*2: -1])
        print(train0.shape, train1.shape, train2.shape, len(items))
        plt.plot(train0[0], train0[1], train1[0], train1[1], train2[0], train2[1])
        plt.legend(["No data-augmentation, No dropout", "No dropout", "Data-augmentation and dropout"])
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training loss")
        plt.axis([0, 200000, 0, 2])
        plt.show()
        print(train2)