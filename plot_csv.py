import csv
import matplotlib
import matplotlib.pyplot as plt
if __name__ == "__main__":
    x1 = []
    y1 = []
    z1 = []

    with open('Non_IID_emnist-E=5_0cluster_fixcarbon30_final.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        i=0
        for row in plots:
            if(i > 0):
                x1.append(float(row[2]))
                y1.append(float(row[1]))
            i = i+1


    x2 = []
    y2 = []
    z2 = []

    with open('Non_IID_emnist-E=5_1cluster_fixcarbon30_final.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x2.append(float(row[2]))
                y2.append(float(row[1]))
            i = i + 1

    x3 = []
    y3 = []
    z3 = []

    with open('Non_IID_emnist-E=5_2cluster_fixcarbon30_final.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x3.append(float(row[2]))
                y3.append(float(row[1]))
            i = i + 1

    x4 = []
    y4 = []
    z4 = []

    with open('Non_IID_emnist-E=5_3cluster_fixcarbon30_final.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x4.append(float(row[2]))
                y4.append(float(row[1]))

            i = i + 1




    # Plot a graph

    matplotlib.use('Agg')
    # Plot Loss curve

    plt.figure()
    plt.title('Non-IID EMNIST')
    # Plot Loss curve


    plt.plot(x1, y1, color='r', label='0cluster(p=1)')
    plt.plot(x2, y2, color='b', label='1cluster(p=0.6)')
    plt.plot(x3, y3, color='g', label='2cluster(p=0.2 , 1)')

    plt.plot(x4, y4, color='k', label='3cluster(p=0.2 , 0.6 , 1)')


    plt.legend()
    plt.ylabel('test accuracy')
    plt.xlabel('Carbon Cost(kg)')
    plt.savefig('save/plot-test_5.png')