import csv
import matplotlib
import matplotlib.pyplot as plt
if __name__ == "__main__":
    x1 = []
    y1 = []
    z1 = []

    with open('BalanceExperiment/Emnist_E=1_alpha=0.01_1cluster(m=1)_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('BalanceExperiment/Emnist_E=1_alpha=0.01_1cluster(m=0.6)_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('BalanceExperiment/Emnist_E=1_alpha=0.01_2cluster(m=0.6 , sd=0.4)_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('BalanceExperiment/Emnist_E=1_alpha=0.01_3cluster(m=0.6 , sd=0.32)_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('BalanceExperiment/Emnist_E=1_alpha=0.01_dcluster(s=2(0.6,0.4) - e=3(0.6,0.32))_200round_feq1_real.csv', 'r') as csvfile:
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

    with open('BalanceExperiment/Emnist_E=1_alpha=0.01_dcluster(s=3(0.6,0.32) - e=2(0.6,0.4))_200round_feq1_real.csv', 'r') as csvfile:
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


    plt.plot(z1, y1, color='r', label='1 cluster m= 1, sd=0(FedAvg)' , linewidth=0.7)
    plt.plot(z2, y2, color='k', label='1 cluster m= 0.6, sd=0', linewidth=0.7)

    plt.plot(z3, y3, color='g', label='2 clusters m= 0.6, sd=0.4', linewidth=0.7)
    plt.plot(z4, y4, color='y', label='3 clusters m= 0.6, sd=0.32', linewidth=0.7)

    plt.plot(z5, y5, color='b', label='clusters dynamic1', linewidth=0.7)
    #plt.plot(z6, y6, color='y', label='clusters dynamic2', linewidth=0.7)

    plt.legend()
    plt.ylabel('test accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('final_plots/alpha=0.01_dynamicincrease_E=1_round.svg')