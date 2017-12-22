# -*- coding: utf-8 -*-
import os
from random import uniform

import matplotlib.pyplot as plt
import numpy as np
import xlrd
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score

_selected_cols = (1, 3, 4, 6, 7, 8)
_ncols = len(_selected_cols)
_ninput = _ncols - 1
_font_path = r"C:\Windows\Fonts\msyh.ttc"
_fname = "作业数据_2017And2016.xls"
_node_type_code = {"input_layer": 0, "hidden_layer": 1, "output_layer": 2}
f = lambda x: 1 / (1 + np.exp(-x))
df = lambda x: x * (1 - x)
rf = lambda x: np.log(x) - np.log(1 - x)
yita = 0.1
MAX_TRIAL = 5000
THRESHOLD = 0.1


def predict(bpnet, l_input, output):
    estimation = bpnet.calculate_output(l_input)
    real = np.round(rf(output[0]))
    estimation = np.round(rf(estimation[0]))
    if real == estimation:
        return True
    else:
        return False


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


def plot_test():
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    z = np.cos(x ** 2)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="$sin(x)$", color="red", linewidth=2)
    plt.plot(x, z, "b--", label="$cos(x^2)$")
    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("PyPlot First Example")
    plt.ylim(-1.2, 1.2)
    plt.legend()
    plt.show()


def check_row(row, ncols=_ncols):
    if row is None:
        return False
    if type(row) != list:
        return False
    if row.__len__() != ncols:
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


def regulate(data, cols):
    nrow = data.__len__()
    for col in cols:
        l_max = -100000
        l_min = 1000000
        for i in np.arange(nrow):
            if data[i][col] > l_max:
                l_max = data[i][col]
            if data[i][col] < l_min:
                l_min = data[i][col]
        delta = l_max - l_min
        for i in np.arange(nrow):
            data[i][col] = (data[i][col] - l_min) / delta


def split_sample(data, ratio):
    if ratio <= 0 or ratio >= 1:
        return None
    from random import randint
    boys = []
    girls = []
    for row in data:
        if row[0] == 1:
            boys.append(row)
        else:
            girls.append(row)
    nboys = len(boys)
    ngirls = len(girls)
    n = int(np.round(nboys * ratio))
    m = int(np.round(ngirls * ratio))
    train_set = []
    test_set = []
    for i in range(n):
        index = randint(0, len(boys) - 1)
        train_set.append(boys[index])
        boys.remove(boys[index])
    for i in range(m):
        index = randint(0, len(girls) - 1)
        train_set.append(girls[index])
        girls.remove(girls[index])
    test_set.extend(boys)
    test_set.extend(girls)
    return train_set, test_set


def plot_bar():
    myfont = font_manager.FontProperties(fname=_font_path)
    data = read_excel()
    male = {"150": 0, "155": 0, "160": 0, "165": 0, "170": 0, "175": 0, "180": 0, "185": 0}
    female = {"150": 0, "155": 0, "160": 0, "165": 0, "170": 0, "175": 0, "180": 0, "185": 0}
    for row in data:
        if row[0] == 1:  # male
            height = row[1]
            index = str(int(height // 5 * 5))
            male[str(index)] = male[str(index)] + 1
        elif row[0] == 0:  # female
            height = row[1]
            index = str(int(height // 5 * 5))
            female[index] = female[str(index)] + 1
    index = np.arange(8)
    bar_width = 0.35
    opacity = 0.4
    rects1 = plt.bar(index, list(male.values()), bar_width, alpha=opacity, color='b', label='Men')
    rects2 = plt.bar(index + bar_width, list(female.values()), bar_width, alpha=opacity, color='r', label='Women')

    plt.xlabel('身高', fontproperties=myfont)
    plt.ylabel('人数', fontproperties=myfont)
    plt.title('男女身高直方图', fontproperties=myfont)
    plt.xticks(index + bar_width, ('150', '155', '160', '165', '170', '175', '180', '185'))
    plt.legend()
    plt.show()


def MLE(data):
    if data is None:
        return
    darray = data
    u_head = np.mean(darray, 0)
    sum = 0
    for vector in darray:
        delta = vector - u_head
        sum += np.linalg.norm(delta) ** 2
    square_delta_head = sum / darray.shape[0]
    print(u_head, square_delta_head)
    return u_head, square_delta_head


def bayes_estimation(data, u0, stf0, stf):
    if data is None:
        return
    darray = data
    sum_ = darray.sum(0)
    estimation = stf0 / (darray.shape[0] * stf0 + stf) * sum_ + stf / (darray.shape[0] * stf0 + stf) * u0
    print(estimation, stf)
    return estimation


def plot_decision_plane(boys, girls):
    boys = np.mat(boys)
    girls = np.mat(girls)
    mb = np.mean(boys, 0)
    mg = np.mean(girls, 0)
    covb = np.cov(boys.T)
    covg = np.cov(girls.T)
    inv_covb = np.linalg.inv(covb)
    inv_covg = np.linalg.inv(covg)
    xs = np.linspace(145, 185, 10000)
    ys = np.linspace(40, 80, 1000)
    threshould = 0.5

    def judge(x, y):
        vec = np.array([x, y])
        result = -1 / 2 * ((vec - mb) * inv_covb * (vec - mb).T - (vec - mg) * inv_covg * (vec - mg).T) \
                 - 1 / 2 * np.log(np.linalg.det(covb) / np.linalg.det(covg)) \
                 + np.log(boys.shape[0] / girls.shape[0])
        return np.abs(result) < threshould

    OK = []
    for x in xs:
        temp = []
        for y in ys:
            if judge(x, y):
                temp.append(y)
        if temp:
            OK.append([x, np.mean(temp)])
    for i in OK:
        print(i)
    color = ['g'] * boys.shape[0] + ['r'] * girls.shape[0]
    corrdinate = np.vstack([boys, girls])
    plt.scatter(list(corrdinate[:, 0]), list(corrdinate[:, 1]), c=color)
    plt.plot([x[0] for x in OK], [x[1] for x in OK], color="blue")
    plt.show()


if __name__ == "__main__":
    data = read_excel()
    do_pretreatment(data)
    boys = []
    girls = []
    for row in data:
        if row[0] == 1:
            boys.append(row)
        else:
            girls.append(row)
    dboys = np.array(boys)
    dgirls = np.array(girls)
    MLE(dboys[:, 1])
    MLE(dboys[:, 2])
    MLE(dgirls[:, 1])
    MLE(dgirls[:, 2])
    bayes_estimation(dboys[:, 1], 170, 50, 100)
    bayes_estimation(dboys[:, 2], 60, 25, 100)
    bayes_estimation(dgirls[:, 1], 160, 25, 100)
    bayes_estimation(dgirls[:, 2], 45, 25, 100)
    plot_decision_plane(dboys[:, 1:3], dgirls[:, 1:3])
    # plot_bar()
