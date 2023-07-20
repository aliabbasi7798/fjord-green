import csv
import matplotlib
import matplotlib.pyplot as plt
if __name__ == "__main__":
    x1 = []
    y1 = []
    z1 = []

    with open('Non_IID_emnist-E=5_1cluster_fixcarbon30.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        i=0
        for row in plots:
            if(i > 0):
                x1.append(float(row[2]))
                y1.append(float(row[1]))
                z1.append(float(row[0]))
            i = i+1


    x2 = []
    y2 = []
    z2 = []

    with open('Non_IID_emnist-E=5_1cluster_fixcarbon30_m=0.8.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x2.append(float(row[2]))
                y2.append(float(row[1]))
                z2.append(float(row[0]))
            i = i + 1

    x3 = []
    y3 = []
    z3 = []

    with open('Non_IID_emnist-E=5_1cluster_fixcarbon30_m=0.4.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x3.append(float(row[2]))
                y3.append(float(row[1]))
                z3.append(float(row[0]))
            i = i + 1

    x4 = []
    y4 = []
    z4 = []

    with open('Non_IID_emnist-E=5_1cluster_r15_m=1.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x4.append(float(row[2]))
                y4.append(float(row[1]))
                z4.append(float(row[0]))

            i = i + 1

    x5 = []
    y5 = []
    z5 = []

    with open('Non_IID_emnist-E=5_1cluster_fixcarbon30_m=0.2.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x5.append(float(row[2]))
                y5.append(float(row[1]))
                z5.append(float(row[0]))

            i = i + 1

    # Plot a graph

    matplotlib.use('Agg')
    # Plot Loss curve

    plt.figure()
    plt.title('Non-IID EMNIST(E=5)')
    # Plot Loss curve


    plt.plot(z1[0:29], y1[0:29], color='r', label='1 cluster m= 0.6 , sd=0')
    #plt.plot(z2[0:29], y2[0:29], color='b', label='1 cluster m= 0.8 , sd=0')
    plt.plot(z3[0:29], y3[0:29], color='g', label='1 cluster m= 0.4 , sd=0')

    plt.plot(z4[0:29], y4[0:29], color='k', label='1 cluster m= 1 , sd=0')
    plt.plot(z5[0:29], y5[0:29], color='m', label='3 cluster m= 0.2 , sd=0')

    plt.legend()
    plt.ylabel('test accuracy')
    plt.xlabel('Communication Round')
    plt.savefig('save/cluster(E=5)_m_r.png')