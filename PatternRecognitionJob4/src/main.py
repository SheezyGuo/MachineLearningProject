from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import xlrd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from random import randint, random

_fname = "作业数据_2017And2016.xls"
_selected_cols = (1, 3, 4)
_ncols = len(_selected_cols)


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


def main():
    data = read_excel()
    do_pretreatment(data)

    info = [info[1:] for info in data]
    dtype = [1 - info[0] for info in data]

    plt.scatter([x[0] for x in info], [x[1] for x in info], c=dtype)
    plt.title("Scatter of excel")
    plt.show()

    kmeans = KMeans(n_clusters=2)
    temp = kmeans.fit_predict(info)

    plt.scatter([x[0] for x in info], [x[1] for x in info], c=temp)
    plt.title("Scatter of excel")
    plt.show()

    count = 0
    for a, b in zip(temp, dtype):
        if a == b:
            count += 1
    print(count / len(dtype))


if __name__ == "__main__":
    main()
