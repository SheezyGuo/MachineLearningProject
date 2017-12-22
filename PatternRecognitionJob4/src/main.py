import os

import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from matplotlib import font_manager

_fname = "作业数据_2017And2016.xls"
_selected_cols = (1, 3, 4)
_ncols = len(_selected_cols)
_font_path = r"C:\Windows\Fonts\msyh.ttc"


def read_excel(fname=_fname):
    cwd = os.getcwd()
    if not cwd.endswith("\\src"):
        return
    fpath = cwd.replace("\\src", "\\resource\\") + fname
    book = xlrd.open_workbook(fpath)
    sheet = book.sheet_by_index(0)
    nrows = sheet.nrows
    ncols = sheet.ncols
    data = []
    row = []
    for i in range(1, nrows):
        for j in _selected_cols:
            row.append(sheet.cell(i, j).value)
        data.append(row)
        row = []
    return data


def check_row(row, ncols=_ncols):
    if row is None:
        return False
    if type(row) != list:
        return False
    if len(row) != ncols:
        return False
    for item in row:
        if item is None or item == '':
            return False
    return True


def do_pretreatment(data):
    if data is None:
        return
    for row in data:
        if check_row(row) is False:
            data.remove(row)


def get_distances(X, model, mode='l2'):
    distances = []
    weights = []
    children = model.children_
    dims = (X.shape[1], 1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1 - c2)
        cc = ((c1W * c1) + (c2W * c2)) / (c1W + c2W)

        X = np.vstack((X, cc.T))

        newChild_id = X.shape[0] - 1

        # How to deal with a higher level cluster merge with lower distance:
        if mode == 'l2':  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist ** 2 + c2Dist ** 2) ** 0.5
            dNew = (d ** 2 + added_dist ** 2) ** 0.5
        elif mode == 'max':  # If the previrous clusters had higher distance, use that one
            dNew = max(d, c1Dist, c2Dist)
        elif mode == 'actual':  # Plot the actual distance.
            dNew = d

        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append(wNew)
    return distances, weights


def main():
    data = read_excel()
    do_pretreatment(data)
    myfont = font_manager.FontProperties(fname=_font_path)
    info = [info[1:] for info in data]
    dtype = [1 - info[0] for info in data]

    plt.scatter([x[0] for x in info], [x[1] for x in info], c=dtype)
    plt.title("男女原分布散点图", fontproperties=myfont)
    plt.show()

    kmeans = KMeans(n_clusters=2)
    temp = kmeans.fit_predict(info)

    plt.scatter([x[0] for x in info], [x[1] for x in info], c=temp)
    plt.title("男女聚类图", fontproperties=myfont)
    plt.show()

    count = 0
    for a, b in zip(temp, dtype):
        if a == b:
            count += 1
    print(count / len(dtype))

    model = AgglomerativeClustering(n_clusters=2, linkage="ward")
    model.fit(info)
    distance, weight = get_distances(np.array(info), model)
    linkage_matrix = np.column_stack([model.children_, distance, weight]).astype(float)
    plt.figure(figsize=(20, 10))
    dendrogram(linkage_matrix)
    plt.title("分级聚类图", fontproperties=myfont)
    plt.show()


if __name__ == "__main__":
    main()
