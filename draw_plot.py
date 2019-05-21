import sklearn.metrics
import matplotlib
# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
# matplotlib.rcParams['figure.figsize'] = (9, 6)

result_dir = './test_result_Zeng'
def ttt():
    data = json.load(open('./test_result/nyt_new_pcnn_one_pred.json', 'r', encoding='utf-8'))
    temp = 0
    for i in range(len(data)):
        if i % 52 == 0 :
            temp = 0
        temp += data[i]['score']


def mean():
    x = []
    y = []
    for model in ['nyt_new_none_cnn_att2', 'nyt_new_none_cnn_att3']:
        x.append(np.load(os.path.join(result_dir, model + '_x' + '.npy')))
        y.append(np.load(os.path.join(result_dir, model + '_y' + '.npy')))
    x = np.array(x).mean(0)
    y = np.array(y).mean(0)
    np.save(os.path.join(result_dir, 'nyt_new_none_cnn_att23' + '_x' + '.npy'), x)
    np.save(os.path.join(result_dir, 'nyt_new_none_cnn_att23' + '_y' + '.npy'), y)



def main():
    marker = ['<', '^', 'v']
    color = ['b', 'orange', 'g']
    marker1 = ['s', 'o', 'd']
    color1 = ['m', 'c', 'r']


    # for model in ['nyt_new_pcnn_att1']:
    #     x = []
    #     y = []
    #     with open('./test_result_Zeng/'+model+'.txt', 'r') as f:
    #         lines = f.readlines()
    #         for temp in lines:
    #             temp_x, temp_y = temp.split()
    #             x.append(float(temp_x))
    #             y.append(float(temp_y))
    #     x = np.array(x)
    #     y = np.array(y)
    #     f1 = (2 * x * y / (x + y + 1e-20)).max()
    #     auc = sklearn.metrics.auc(x=y, y=x)
    #     plt.plot(y, x, lw=2, label='./result_Zeng/'+model)
    #     print('./result_Zeng/'+model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(f1))
    #     print('    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(x[100], x[200], x[300],
    #                                                                     (x[100] + x[200] + x[300]) / 3))

    for i, model in enumerate(['MultiR', 'MIML', 'Mintz']):
        x = []
        y = []
        with open('./result_baseline/'+model+'.txt', 'r') as f:
            lines = f.readlines()
            for temp in lines:
                temp_x, temp_y = temp.split()
                x.append(float(temp_x))
                y.append(float(temp_y))
        x = np.array(x)
        y = np.array(y)
        f1 = (2 * x * y / (x + y + 1e-20)).max()
        auc = sklearn.metrics.auc(x=x, y=y)
        plt.plot(x, y, lw=1, label=model, marker=marker[i % len(marker)], markevery=200, markersize=5, color=color[i % len(color)])
        print(model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(f1))
        print('    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[100], y[200], y[300],
                                                                        (y[100] + y[200] + y[300]) / 3))

    # models = set(model[:-6] for model in os.listdir('./test_result') if model.find('.npy') != -1)
    models = sys.argv[1:]
    for i, model in enumerate(models):
        x = np.load(os.path.join(result_dir, model +'_x' + '.npy')) 
        y = np.load(os.path.join(result_dir, model + '_y' + '.npy'))
        f1 = (2 * x * y / (x + y + 1e-20)).max()
        auc = sklearn.metrics.auc(x=x, y=y)
        #plt.plot(x, y, lw=2, label=model + '-auc='+str(auc))
        plt.plot(x, y, lw=1, label=model, marker=marker1[i % len(marker1)], markevery=200, markersize=5, color=color1[i % len(color1)])
        print(model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(f1))
        print('    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[100], y[200], y[300], (y[100] + y[200] + y[300]) / 3))
       
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='dashdot')
    plt.savefig(os.path.join(result_dir, 'pr_curve'), dpi = 300)

if __name__ == "__main__":
    main()
    # mean()
