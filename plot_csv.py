import csv
import matplotlib
import matplotlib.pyplot as plt
if __name__ == "__main__":
    x1 = []
    y1 = []
    z1 = []

    with open('Non_IID_emnist-E=1_1cluster_fix100_m=1_feq5.csv', 'r') as csvfile:
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

    with open('Non_IID_emnist-E=1_2cluster(sd=0.32)_fixcarbon15_feq5.csv', 'r') as csvfile:
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

    with open('Non_IID_emnist-E=1_1cluster_fix100_m=0.6_feq5.csv', 'r') as csvfile:
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

    with open('Non_IID_emnist-E=1_3cluster_fixcarbon15_feq5.csv', 'r') as csvfile:
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

    with open('Non_IID_emnist-E=1_3cluster(sd=0.16)_fixcarbon15_feq5.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x5.append(float(row[2]))
                y5.append(float(row[1]))
                z5.append(float(row[0]))

            i = i + 1
    x6 = []
    y6 = []
    z6 = []

    with open('Non_IID_emnist-E=1_2cluster(sd=0.08)_fixcarbon15_feq5.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x6.append(float(row[2]))
                y6.append(float(row[1]))
                z6.append(float(row[0]))

            i = i + 1
    # Plot a graph

    matplotlib.use('Agg')
    # Plot Loss curve

    plt.figure()
    plt.title('Non-IID EMNIST(E=1)')
    # Plot Loss curve


    plt.plot(x1[0:5], y1[0:5], color='r', label='1 cluster m= 1 , sd=0(FedAvg)')
    plt.plot(x3[0:12], y3[0:12], color='g', label='1 cluster m= 0.6 , sd=0')

    plt.plot(x2, y2, color='b', label='2 cluster m= 0.6 , sd=0.32')
    plt.plot(x6, y6, color='y', label='2 cluster m= 0.6 , sd=0.08')

    plt.plot(x4, y4, color='k', label='3 cluster m= 0.6 , sd=0.32')
    plt.plot(x5, y5, color='m', label='3 cluster m= 0.6 , sd=0.08')

    plt.legend()
    plt.ylabel('test accuracy')
    plt.xlabel('Carbon Cost')
    plt.savefig('save/cluster(E=1)_m_c_4.png')