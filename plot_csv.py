import csv
import matplotlib
import matplotlib.pyplot as plt
if __name__ == "__main__":
    x1 = []
    y1 = []
    z1 = []

    with open('new_experiments/Emnist_E=1_alpha=0.01_1cluster(m=1,sd=0)_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('new_experiments/Emnist_E=1_alpha=0.01_1cluster(m=0.6,sd=0)_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('new_experiments/Emnist_E=1_alpha=0.01_2cluster(m=0.6, sd=0.4)_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('new_experiments/Emnist_E=1_alpha=0.01_3cluster(m=0.6,sd=0.32)_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('new_experiments/Emnist_E=1_alpha=0.01_1cluster(s=0.6,0.4,2 , 0,e=0.6,0.4,3 )_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('new_experiments/Emnist_E=1_alpha=0.01_1cluster(s=0.6,0.4 , 0,e=0.6)_200round_feq1_real.csv', 'r') as csvfile:
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
    #plt.title('Non-IID EMNIST(alpha = 0.01)(E=5)_real intensity')
    # Plot Loss curve


    #plt.plot(x1, y1, color='r', label='1 cluster m= 1, sd=0(FedAvg)' , linewidth=0.7)
    #plt.plot(z2, y2, color='k', label='1 cluster m= 0.6, sd=0', linewidth=0.7)

    plt.plot(x3, y3, color='g', label='2 clusters m= 0.6, sd=0.4', linewidth=0.7)
    plt.plot(x4, y4, color='r', label='3 clusters m= 0.6, sd=0.32', linewidth=0.7)

    plt.plot(x5, y5, color='b', label='clusters dynamic s=0.6,0.4,2, e=0.6,0.32,3', linewidth=0.7)
    plt.plot(x6, y6, color='y', label='1 clusters dynamic s=0.6,0.4, e=0.6', linewidth=0.7)

    plt.legend()
    plt.ylabel('test accuracy')
    plt.xlabel('Carbon cost')
    plt.savefig('final_plots/alpha=0.01_dynamic3_E=1_cost.svg')