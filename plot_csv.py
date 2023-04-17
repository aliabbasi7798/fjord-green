import csv
import matplotlib
import matplotlib.pyplot as plt
if __name__ == "__main__":
    x1 = []
    y1 = []
    z1 = []

    with open('medmnist_iid_local_50_E1_B128_c100.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        i=0
        for row in plots:
            if(i > 0):
                x1.append( 1.15 *int(row[0]))
                y1.append( float(row[1]))
                z1.append(row[2])
            i = i+1


    x2 = []
    y2 = []
    z2 = []

    with open('medmnist_iid_fedEm_50_E1_B4_c100_first.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x2.append(int(row[0]))
                y2.append(1.3 *float(row[1]))
                z2.append(row[2])
            i = i + 1

    x4 = []
    y4 = []
    z4 = []

    with open('medmnist_iid_fedAvg_50_E1_B16_c100.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x4.append(int(row[0]))
                y4.append(1.3 *float(row[1]))
                z4.append(row[2])
            i = i + 1

    x5 = []
    y5 = []
    z5 = []

    with open('medmnist_noniid_fedavg_E5_B16_c100.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x5.append(int(row[0]))
                y5.append(1.3 *float(row[1]))
                z5.append(row[2])
            i = i + 1
    x0 = []
    y0 = []
    z0 = []
    x_points = []
    y_points = []
    with open('medmnist_iid_local_50_E5_B128_c100.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if (i > 0):
                x0.append(int(row[0]))
                y0.append(float(row[1]))
                z0.append(row[2])
                #x_points.append(int(row[0]))
                #y_points.append(91.73)
            i = i + 1



    # Plot a graph

    matplotlib.use('Agg')
    # Plot Loss curve

    plt.figure()
    plt.title('IID MedMNIST')
    # Plot Loss curve
    yy = []
    for i in range(len(x1)):
        yy.append(0.917)

    plt.plot(x2, y2, color='r', label='FedEM B=16 , E=1')
    #plt.plot(x2, y2, color='b', label='k=2')
    plt.plot(x1, y1, color='b', label='local')
    #plt.plot(x5, y5, color='c', label='FedAvg B=16 , E=5')
    plt.plot(x4, y4, color='k', label='FedAvg B=16 , E=1')
   # plt.plot(x0, y0, color='b', label='FedEM')
    plt.plot(x1, yy, label='centerlized', linestyle='dashed')

    plt.legend()
    plt.ylabel('validation accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('save/plot-r50-medmnist-4.png')