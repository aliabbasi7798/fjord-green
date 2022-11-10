import csv
import matplotlib
import matplotlib.pyplot as plt
if __name__ == "__main__":
    x1 = []
    y1 = []
    z1 = []

    with open('do(k=1).csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        i=0
        for row in plots:
            if(i > 0):
                x1.append(int(row[0]))
                y1.append(float(row[1]))
                z1.append(row[2])
            i = i+1
    x2 = []
    y2 = []
    z2 = []

    with open('do(k=2).csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x2.append(int(row[0]))
                y2.append(float(row[1]))
                z2.append(row[2])
            i = i + 1

    x4 = []
    y4 = []
    z4 = []

    with open('do(k=4).csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x4.append(int(row[0]))
                y4.append(float(row[1]))
                z4.append(row[2])
            i = i + 1

    x5 = []
    y5 = []
    z5 = []

    with open('do(k=5).csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x5.append(int(row[0]))
                y5.append(float(row[1]))
                z5.append(row[2])
            i = i + 1
    x0 = []
    y0 = []
    z0 = []

    with open('do(k=avg).csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x0.append(int(row[0]))
                y0.append(float(row[1]))
                z0.append(row[2])
            i = i + 1

    matplotlib.use('Agg')
    # Plot Loss curve

    plt.figure()
    plt.title('Train accuracy')
    # Plot Loss curve
    plt.plot(x1, y1, color='r', label='Fjord , k=1')
    #plt.plot(x2, y2, color='b', label='k=2')
    #plt.plot(x3, y3, color='g', label='k=3')
    #plt.plot(x4, y4, color='c', label='k=4')
    plt.plot(x5, y5, color='k', label='Fjord , k=5')
    #plt.plot(x0, y0, color='g', label='FedAvg')

    plt.legend()
    plt.ylabel('train accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('save/plot(k)6.png')