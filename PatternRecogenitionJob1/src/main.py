# -*- coding: utf-8 -*-
import os
from random import uniform

import matplotlib.pyplot as plt
import numpy as np
import xlrd
from matplotlib import font_manager

_selected_cols = (1, 3, 4, 6, 7, 8)
_ncols = _selected_cols.__len__()
_ninput = _ncols - 1
_font_path = r"C:\Windows\Fonts\msyh.ttc"
_fname = "作业数据_2017And2016.xls"
_node_type_code = {"input_layer": 0, "hidden_layer": 1, "output_layer": 2}
f = lambda x: 1 / (1 + np.exp(-x))
df = lambda x: f(x) * (1 - f(x))
yita = 0.08
MAX_TRIAL = 5000
THRESHOLD = 0.1


class NetworkNode(object):
    def __init__(self):
        self.type_code = -1  # 节点类型code
        self.serial_num = 0  # 节点序列号
        self.prior_node_list = []  # 前层节点列表
        self.inferior_node_list = []  # 后层节点列表
        self.weight_list = []  # 前层节点到该节点的权重列表
        self.weight_increment_list = []  # 暂存增量列表
        self.delta = 0  # 该节点的delta值
        self.output = 0  # 所有前层节点的输出乘以权重再求和
        # self.bias = 0  # 计算netj中的常数项

    def set_type_code(self, type_name):
        if type_name.find("hidden_layer") >= 0:
            type_name = "hidden_layer"
        if type_name not in _node_type_code.keys():
            return False
        self.type_code = _node_type_code[type_name]

    def set_serial_num(self, serial_num):
        self.serial_num = serial_num

    def set_prior_node_list(self, prior_node_list):
        self.prior_node_list = prior_node_list

    def set_inferior_node_list(self, inferior_node_list):
        self.inferior_node_list = inferior_node_list

    def update_delta(self, estimation_k=None):
        if self.type_code is _node_type_code["output_layer"]:
            self.delta = -(estimation_k - self.output) * df(self.output)
        else:
            l_sum = 0
            for inode in np.arange(self.inferior_node_list.__len__()):
                node = self.inferior_node_list[inode]
                inferior_delta = node.delta
                inferior_weight = node.weight_list[self.serial_num]
                l_sum += inferior_delta * inferior_weight
            self.delta = l_sum * df(self.output)

    def update_weight_list(self, weight_list):
        if type(weight_list) is not list:
            return False
        self.weight_list = weight_list

    def update_output(self):
        l_sum = 0
        for i in np.arange(self.prior_node_list.__len__()):
            l_sum += self.prior_node_list[i].output * self.weight_list[i]
        output = f(l_sum)
        self.output = output


class BPNetwork(object):
    _node_num_of_each_layer = ()
    _name_of_each_layer = []
    _nlayers = 0
    _layer_nodes = {}

    def get_layer_nodes(self):
        return self._layer_nodes

    def set_node_num_of_each_layer(self, layer_num):
        if layer_num is None:
            return False
        if type(layer_num) is not tuple:
            return False
        self._node_num_of_each_layer = layer_num
        self._nlayers = self._node_num_of_each_layer.__len__()
        return True

    def set_name_of_each_layer(self):
        self._name_of_each_layer.append("input_layer")
        for index in np.arange(1, self._nlayers - 1):
            self._name_of_each_layer.append("hidden_layer" + str(index))
        self._name_of_each_layer.append("output_layer")

    def init_network(self, node_num_of_each_layer=(_ninput, 5, 4, 3, 2, 1)):
        self._nlayers = node_num_of_each_layer.__len__()
        self._node_num_of_each_layer = node_num_of_each_layer
        if self._nlayers < 2:
            return False
        self.set_name_of_each_layer()
        self._layer_nodes = {}
        for index in np.arange(self._nlayers):
            layer_name = self._name_of_each_layer[index]
            node_num = self._node_num_of_each_layer[index]
            nodes = []
            for i in np.arange(node_num):
                node = NetworkNode()
                node.set_type_code(layer_name)
                node.serial_num = i
                nodes.append(node)
            self._layer_nodes[layer_name] = nodes

        for index in np.arange(self._nlayers):
            layer_name = self._name_of_each_layer[index]
            for inode in np.arange(self._layer_nodes[layer_name].__len__()):
                node = self._layer_nodes[layer_name][inode]
                if node.type_code is not _node_type_code["input_layer"]:
                    node.set_prior_node_list(self._layer_nodes[self._name_of_each_layer[index - 1]])
                    weights = []
                    for i in np.arange(self._node_num_of_each_layer[index - 1]):
                        weights.append(uniform(-1, 1))
                    node.update_weight_list(weights)
                if node.type_code is not _node_type_code["output_layer"]:
                    node.set_inferior_node_list(self._layer_nodes[self._name_of_each_layer[index + 1]])
        self.__set_zero()

    def fill_input_layer(self, l_input):
        if type(l_input) is not tuple:
            return False
        if l_input.__len__() is not _ninput:
            return False
        for i in np.arange(_ninput):
            self._layer_nodes["input_layer"][i].output = l_input[i]

    def __init__(self, attr_tuple=None):
        if type(attr_tuple) is not tuple:
            pass
        if attr_tuple is None:
            self.init_network()
        else:
            self.init_network(attr_tuple)

    def calculate_output(self, l_input):
        self.fill_input_layer(l_input)
        for layer_name in self._name_of_each_layer[1:]:
            for inode in np.arange(self._layer_nodes[layer_name].__len__()):
                node = self._layer_nodes[layer_name][inode]
                node.update_output()
        output_list = []
        for index in np.arange(self._layer_nodes["output_layer"].__len__()):
            output_list.append(self._layer_nodes["output_layer"][index].output)
        return tuple(output_list)

    def calculate_single_sample_error(self, l_input, output_value):
        if type(l_input) is not tuple:
            return False
        if l_input.__len__() is not _ninput:
            return False
        if type(output_value) is not tuple:
            return False
        if output_value.__len__() is not self._layer_nodes["output_layer"].__len__():
            return False
        self.calculate_output(l_input)
        foutput_list = []
        for output in output_value:
            foutput_list.append(f(output))
        output_value = tuple(foutput_list)
        error = 0
        for i in np.arange(self._layer_nodes["output_layer"].__len__()):
            error += (self._layer_nodes["output_layer"][i].output - output_value[i]) ** 2
        error /= 2
        return error

    def __bp_calculate_delta(self, estimation):
        for index in np.arange(self._nlayers - 1, -1, -1):
            layer_name = self._name_of_each_layer[index]
            node_num = self._node_num_of_each_layer[index]
            for inode in np.arange(node_num):
                node = self._layer_nodes[layer_name][inode]
                if layer_name is "output_layer":
                    node.update_delta(estimation[inode])
                else:
                    node.update_delta()

    def __set_zero(self):
        for index in np.arange(self._nlayers - 1, 0, -1):
            layer_name = self._name_of_each_layer[index]
            node_num = self._node_num_of_each_layer[index]
            for inode in np.arange(node_num):
                node = self._layer_nodes[layer_name][inode]
                if node.weight_increment_list.__eq__([]):
                    for prior_node_index in np.arange(node.prior_node_list.__len__()):
                        node.weight_increment_list.append(0)
                else:
                    for prior_node_index in np.arange(node.prior_node_list.__len__()):
                        node.weight_increment_list[prior_node_index] = 0

    def __bp_calculate_weight_increment(self):
        for index in np.arange(self._nlayers - 1, 0, -1):
            layer_name = self._name_of_each_layer[index]
            node_num = self._node_num_of_each_layer[index]
            for inode in np.arange(node_num):
                node = self._layer_nodes[layer_name][inode]
                for prior_node_index in np.arange(node.prior_node_list.__len__()):
                    prior_node = node.prior_node_list[prior_node_index]
                    weight_increment = -1 * yita * node.delta * prior_node.output
                    node.weight_increment_list[prior_node_index] += weight_increment

    def calculate_mean_increment(self, sample_num):
        for index in np.arange(self._nlayers - 1, 0, -1):
            layer_name = self._name_of_each_layer[index]
            node_num = self._node_num_of_each_layer[index]
            for inode in np.arange(node_num):
                node = self._layer_nodes[layer_name][inode]
                for i in np.arange(node.weight_increment_list.__len__()):
                    node.weight_increment_list[i] /= sample_num

    def add_weight_increment(self):
        for index in np.arange(self._nlayers - 1, 0, -1):
            layer_name = self._name_of_each_layer[index]
            node_num = self._node_num_of_each_layer[index]
            for inode in np.arange(node_num):
                node = self._layer_nodes[layer_name][inode]
                for i in np.arange(node.weight_list.__len__()):
                    node.weight_list[i] += node.weight_increment_list[i]
        self.__set_zero()

    def bp_adjust(self, estimation):
        self.__bp_calculate_delta(estimation)
        self.__bp_calculate_weight_increment()


def train(bpnet, max_trial=MAX_TRIAL, threshold=THRESHOLD):
    count = 0
    data = read_excel()
    do_pretreatment(data)
    regulate(data, (1, 2))
    while count < max_trial:
        error_sum = 0
        for row in data:
            input_value = tuple(row[1:])
            output_value = tuple(row[-1:])
            error = bpnet.calculate_single_sample_error(input_value, output_value)
            error_sum += error
            bpnet.bp_adjust(output_value)
        bpnet.calculate_mean_increment(data.__len__())
        bpnet.add_weight_increment()
        if error_sum <= threshold:
            print("Finished before try out" + "#" * 20)
            break
        count += 1
    return bpnet


def predict(bpnet, l_input, output):
    estimation = bpnet.calculate_output(l_input)
    print(str.format("Real:{} Estimation:{}", output[0], estimation[0]))


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
    if type(row) is not list:
        return False
    if row.__len__() is not ncols:
        return False
    for item in row:
        if item is None or item is '':
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


def split_sample(data):
    boys = []
    girls = []
    for row in data:
        if row[0].__eq__(1):
            boys.append(row)
        else:
            girls.append(row)



def plot_bar():
    myfont = font_manager.FontProperties(fname=_font_path)
    data = read_excel()
    male = {"150": 0, "155": 0, "160": 0, "165": 0, "170": 0, "175": 0, "180": 0, "185": 0}
    female = {"150": 0, "155": 0, "160": 0, "165": 0, "170": 0, "175": 0, "180": 0, "185": 0}
    for row in data:
        # print(type(row[0]))
        if row[0] == 1:  # male
            # print("men")
            height = row[1]
            index = str(int(height // 5 * 5))
            male[str(index)] = male[str(index)] + 1
        elif row[0] == 0:  # female
            height = row[1]
            index = str(int(height // 5 * 5))
            female[index] = female[str(index)] + 1
            # print("women")
    index = np.arange(8)
    bar_width = 0.35
    opacity = 0.4
    rects1 = plt.bar(index, list(male.values()), bar_width, alpha=opacity, color='b', label='Men')
    rects2 = plt.bar(index + bar_width, list(female.values()), bar_width, alpha=opacity, color='r', label='Women')

    plt.xlabel('身高', fontproperties=myfont)
    plt.ylabel('人数', fontproperties=myfont)
    plt.title('男女身高直方图', fontproperties=myfont)
    plt.xticks(index + bar_width, ('150', '155', '160', '165', '170', '175', '180', '185'))
    # plt.ylim(0, 40)
    plt.legend()
    # plt.tight_layout()
    plt.show()


def plot_decision_plane():
    pass


def main():
    data = read_excel()
    do_pretreatment(data)
    regulate(data, (1, 2))
    bpnet = BPNetwork()
    train(bpnet, 500, 0.5)
    # train(bpnet)
    for row in data:
        input_value = tuple(row[1:])
        output_value = tuple(row[-1:])
        foutput_list = []
        for output in output_value:
            foutput_list.append(f(output))
        output_value = tuple(foutput_list)
        predict(bpnet, input_value, output_value)


# def svm():
#     from sklearn import


if __name__ == "__main__":
    main()
