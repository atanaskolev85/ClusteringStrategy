import numpy as np
import pandas as pd
import random
import math
from datetime import datetime
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn import mixture
from sklearn.cluster import HDBSCAN
from sklearn.cluster import Birch

# GLOBALS
export_path = '/Users/dmitrievsky/Library/Containers/com.isaacmarovitz.Whisky/Bottles/54CFA88F-36A3-47F7-915A-D09B24E89192/drive_c/Program Files/MetaTrader 5/MQL5/Include/'
SYMBOL = 'EURUSD'

def get_prices() -> pd.DataFrame:
    p = pd.read_csv('files/EURUSD_H1.csv', sep='\s+')
    pFixed = pd.DataFrame(columns=['time', 'close'])
    pFixed['time'] = p['<DATE>'] + ' ' + p['<TIME>']
    pFixed['time'] = pd.to_datetime(pFixed['time'], format='mixed')
    pFixed['close'] = p['<CLOSE>']
    pFixed.set_index('time', inplace=True)
    pFixed.index = pd.to_datetime(pFixed.index, unit='s')
    pFixed = pFixed.dropna()
    pFixedC = pFixed.copy()

    count = 0
    for i in PERIODS:
        pFixed[str(count)] = pFixedC.rolling(i).mean() - pFixedC
        count += 1

    for i in PERIODS_META:
        pFixed[str(count)+'std'] = pFixedC.rolling(i).std()
        count += 1

    return pFixed.dropna()

def get_labels(dataset, min = 1, max = 25):
    labels = []
    meta_labels = []
    for i in range(dataset.shape[0]-max):
        rand = random.randint(min, max)
        curr_pr = dataset['close'].iloc[i]
        future_pr = dataset['close'].iloc[i + rand]

        if future_pr < curr_pr:
            labels.append(1.0)
            if future_pr + MARKUP < curr_pr:
                meta_labels.append(1.0)
            else:
                meta_labels.append(0.0)
        elif future_pr > curr_pr:
            labels.append(0.0)
            if future_pr - MARKUP > curr_pr:
                meta_labels.append(1.0)
            else:
                meta_labels.append(0.0)
        else:
            labels.append(2.0)
            meta_labels.append(0.0)
        
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset['meta_labels'] = meta_labels
    dataset = dataset.dropna()
    dataset = dataset.drop(
        dataset[dataset.labels == 2.0].index)
    
    return dataset

def tester(dataset: pd.DataFrame, plot = False):
    last_deal = int(2)
    last_price = 0.0
    report = [0.0]
    chart = [0.0]
    line = 0
    line2 = 0

    indexes = pd.DatetimeIndex(dataset.index)
    labels = dataset['labels'].to_numpy()
    metalabels = dataset['meta_labels'].to_numpy()
    close = dataset['close'].to_numpy()

    for i in range(dataset.shape[0]):
        if indexes[i] <= FORWARD:
            line = len(report)
        if indexes[i] <= BACKWARD:
            line2 = len(report)

        pred = labels[i]
        pr = close[i]
        pred_meta = metalabels[i] # 1 = allow trades

        if last_deal == 2 and pred_meta==1:
            last_price = pr
            last_deal = 0 if pred <= 0.5 else 1
            continue

        if last_deal == 0 and pred > 0.5 and pred_meta == 1:
            last_deal = 2
            report.append(report[-1] - MARKUP + (pr - last_price))
            chart.append(chart[-1] + (pr - last_price))
            continue

        if last_deal == 1 and pred < 0.5 and pred_meta==1:
            last_deal = 2
            report.append(report[-1] - MARKUP + (last_price - pr))
            chart.append(chart[-1] + (pr - last_price))

    y = np.array(report).reshape(-1, 1)
    X = np.arange(len(report)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y)

    l = lr.coef_
    if l >= 0:
        l = 1
    else:
        l = -1

    if(plot):
        plt.plot(report)
        plt.plot(chart)
        plt.axvline(x = line, color='purple', ls=':', lw=1, label='OOS')
        plt.axvline(x = line2, color='red', ls=':', lw=1, label='OOS2')
        plt.plot(lr.predict(X))
        plt.title("Strategy performance R^2 " + str(format(lr.score(X, y) * l,".2f")))
        plt.xlabel("the number of trades")
        plt.ylabel("cumulative profit in pips")
        plt.show()

    return lr.score(X, y) * l

def test_model(result: list, plt = False):
    pr_tst = get_prices()
    X = pr_tst[pr_tst.columns[1:]]
    X_meta = X.copy()
    X = X.loc[:, ~X.columns.str.contains('std')]
    X_meta = X_meta.loc[:, X_meta.columns.str.contains('std')]

    pr_tst['labels'] = result[0].predict_proba(X)[:,1]
    pr_tst['meta_labels'] = result[1].predict_proba(X_meta)[:,1]
    pr_tst['labels'] = pr_tst['labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    pr_tst['meta_labels'] = pr_tst['meta_labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    return tester(pr_tst, plot=plt)
    
def meta_learner(models_number: int, 
                 iterations: int, 
                 depth: int, 
                 bad_samples_fraction: float, 
                 n_clusters: int, 
                 algorithm: int) -> pd.DataFrame:
    
    dataset = get_labels(get_prices())
    data = dataset[(dataset.index < FORWARD) & (dataset.index > BACKWARD)].copy()
    X = data[data.columns[1:-2]]
    X = X.loc[:, ~X.columns.str.contains('std')]
    meta_X = data.loc[:, data.columns.str.contains('std')]
    y = data['labels']

    B_S_B = pd.DatetimeIndex([])
    for i in range(models_number):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size = 0.5, test_size = 0.5, shuffle = True)
        
        # learn debias model with train and validation subsets
        meta_m = CatBoostClassifier(iterations = iterations,
                                depth = depth,
                                custom_loss = ['Accuracy'],
                                eval_metric = 'Accuracy',
                                verbose = False,
                                use_best_model = True)
        
        meta_m.fit(X_train, y_train, eval_set = (X_val, y_val), plot = False)
        
        coreset = X.copy()
        coreset['labels'] = y
        coreset['labels_pred'] = meta_m.predict_proba(X)[:, 1]
        coreset['labels_pred'] = coreset['labels_pred'].apply(lambda x: 0 if x < 0.5 else 1)
        
        # add bad samples of this iteration (bad labels indices)
        diff_negatives = coreset['labels'] != coreset['labels_pred']
        B_S_B = B_S_B.append(diff_negatives[diff_negatives == True].index)

    to_mark = B_S_B.value_counts()
    marked_idx = to_mark[to_mark > to_mark.mean() * bad_samples_fraction].index
    data.loc[data.index.isin(marked_idx), 'meta_labels'] = 0.0


    if algorithm==0:
        data['clusters'] = KMeans(n_clusters=n_clusters).fit(meta_X).labels_
    elif algorithm==1:
        data['clusters'] = AffinityPropagation().fit(meta_X).predict(meta_X)
    elif algorithm==2:
        data['clusters'] = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0).fit_predict(meta_X)
    elif algorithm==3:
        data['clusters'] = MeanShift().fit_predict(meta_X)
    elif algorithm==4:
        data['clusters'] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(meta_X)
    elif algorithm==5:
        data['clusters'] = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full').fit(meta_X).predict(meta_X)
    elif algorithm==6:
        data['clusters'] = HDBSCAN(min_cluster_size=150).fit_predict(meta_X)
    elif algorithm==7:
        data['clusters'] = Birch(threshold=0.001, n_clusters=n_clusters).fit_predict(meta_X)
    return data[data.columns[1:]]

def fit_final_models(dataset) -> list:
    # features for model\meta models. We learn main model only on filtered labels 
    X, X_meta = dataset[dataset['meta_labels']==1], dataset[dataset.columns[:-3]]
    X = X[X.columns[:-3]]
    X = X.loc[:, ~X.columns.str.contains('std')]
    X_meta = X_meta.loc[:, X_meta.columns.str.contains('std')]
    
    # labels for model\meta models
    y, y_meta = dataset[dataset['meta_labels']==1], dataset[dataset.columns[-1]]
    y = y[y.columns[-3]]
    
    y = y.astype('int16')
    y_meta = y_meta.astype('int16')

    # train\test split
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.8, test_size=0.2, shuffle=True)
    
    train_X_m, test_X_m, train_y_m, test_y_m = train_test_split(
        X_meta, y_meta, train_size=0.8, test_size=0.2, shuffle=True)

    # learn main model with train and validation subsets
    model = CatBoostClassifier(iterations=200,
                               custom_loss=['Accuracy'],
                               eval_metric='Accuracy',
                               verbose=False,
                               use_best_model=True,
                               task_type='CPU')
    model.fit(train_X, train_y, eval_set=(test_X, test_y),
              early_stopping_rounds=25, plot=False)
    
    # learn meta model with train and validation subsets
    meta_model = CatBoostClassifier(iterations=100,
                                    custom_loss=['F1'],
                                    eval_metric='F1',
                                    verbose=False,
                                    use_best_model=True,
                                    task_type='CPU')
    meta_model.fit(train_X_m, train_y_m, eval_set=(test_X_m, test_y_m),
              early_stopping_rounds=15, plot=False)
    
    R2 = test_model([model, meta_model])
    if math.isnan(R2):
        R2 = -1.0
        print('R2 is fixed to -1.0')
    print('R2: ' + str(R2))
    return [R2, model, meta_model]

def export_model_to_ONNX(model, model_number):
    model[1].save_model(
    export_path +'catmodel' + str(model_number) +'.onnx',
    format="onnx",
    export_parameters={
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_doc_string': 'main model',
        'onnx_graph_name': 'CatBoostModel_main'
    },
    pool=None)

    model[2].save_model(
    export_path + 'catmodel_m' + str(model_number) +'.onnx',
    format="onnx",
    export_parameters={
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_doc_string': 'meta model',
        'onnx_graph_name': 'CatBoostModel_meta'
    },
    pool=None)
    
    code = '#include <Math\Stat\Math.mqh>'
    code += '\n'
    code += '#resource "catmodel'+str(model_number)+'.onnx" as uchar ExtModel[]'
    code += '\n'
    code += '#resource "catmodel_m'+str(model_number)+'.onnx" as uchar ExtModel2[]'
    code += '\n'
    code += 'int Periods' + '[' + str(len(PERIODS)) + \
        '] = {' + ','.join(map(str, PERIODS)) + '};'
    code += '\n\n'
    code += 'int Periods_m' + '[' + str(len(PERIODS_META)) + \
        '] = {' + ','.join(map(str, PERIODS_META)) + '};'
    code += '\n\n'

    # get features
    code += 'void fill_arays' + '( double &features[]) {\n'
    code += '   double pr[], ret[];\n'
    code += '   ArrayResize(ret, 1);\n'
    code += '   for(int i=ArraySize(Periods'')-1; i>=0; i--) {\n'
    code += '       CopyClose(NULL,PERIOD_H1,1,Periods''[i],pr);\n'
    code += '       ret[0] = MathMean(pr) - pr[Periods[i]-1];\n'
    code += '       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }\n'
    code += '   ArraySetAsSeries(features, true);\n'
    code += '}\n\n'

    # get features
    code += 'void fill_arays_m' + '( double &features[]) {\n'
    code += '   double pr[], ret[];\n'
    code += '   ArrayResize(ret, 1);\n'
    code += '   for(int i=ArraySize(Periods_m'')-1; i>=0; i--) {\n'
    code += '       CopyClose(NULL,PERIOD_H1,1,Periods_m''[i],pr);\n'
    code += '       ret[0] = MathStandardDeviation(pr);\n'
    code += '       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }\n'
    code += '   ArraySetAsSeries(features, true);\n'
    code += '}\n\n'

    file = open(export_path + str(SYMBOL) + ' ONNX include' + ' ' + str(model_number) + '.mqh', "w")
    file.write(code)

    file.close()
    print('The file ' + 'ONNX include' + '.mqh ' + 'has been written to disk')

def test_all_models(result: list):
    pr_tst = get_prices()
    X = pr_tst[pr_tst.columns[1:]]
    pr_tst['labels'] = 0.5
    pr_tst['meta_labels'] = 0.5
    

    for i in range(len(result)):
        pr_tst['labels'] += result[i][1].predict_proba(X)[:,1]
        pr_tst['meta_labels'] += result[i][2].predict_proba(X)[:,1]

    pr_tst['labels'] = pr_tst['labels'] / (len(result)+1)
    pr_tst['meta_labels'] = pr_tst['meta_labels'] / (len(result)+1)
    pr_tst['labels'] = pr_tst['labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    pr_tst['meta_labels'] = pr_tst['meta_labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)

    return tester(pr_tst, plot=plt)

# HYPERPARAMS
MARKUP = 0.00010
PERIODS = [i for i in range(10, 200, 10)]
PERIODS_META = [20]
BACKWARD = datetime(2010, 1, 1)
FORWARD = datetime(2020, 1, 1)
N_CLUSTERS = 10

# LEARNING LOOP
models = []
for i in range(10):
    data = meta_learner(5, 25, 2, 0.9, n_clusters=N_CLUSTERS, algorithm=4)
    for clust in data['clusters'].unique():
        print(f'Iteration: {i}, Cluster: {clust}')
        filtered_data = data.copy()
        filtered_data['clusters'] = filtered_data['clusters'].apply(lambda x: 1 if x == clust else 0)
        models.append(fit_final_models(filtered_data))
        
# TESTING & EXPORT
models.sort(key=lambda x: x[0])
test_model(models[-1][1:], plt=True)
# export_model_to_ONNX(models[-1], "clusters")
models[-1][2].get_best_score()

