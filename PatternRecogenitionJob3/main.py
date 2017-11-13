import sklearn
import numpy as np
from random import randint, random


class ChromosomeWidthError(Exception):
    pass


class IndexOutOfRangeError(Exception):
    pass


class Single:
    __chromosome_width = 6
    chromosome = []

    def get_chromosome_width(self):
        return self.__chromosome_width

    def __init__(self, str_crm="000000"):
        if type(str_crm) is not str or len(str_crm) is not self.__chromosome_width:
            raise ChromosomeWidthError("Wrong width num")
        for c in str_crm:
            self.chromosome.append(int(c))

    def get_random_chromosome(self):
        str_crm = ""
        for i in range(self.__chromosome_width):
            str_crm += str(randint(0, 1))
        return str_crm


class Groups:
    __groups = []
    __recombine_digit = (randint(0, Single().get_chromosome_width() / 2),
                         randint(Single().get_chromosome_width() / 2, Single().get_chromosome_width() - 1))
    __mutation_probability = 0.001

    def __init__(self, group_num=10):
        if type(group_num) is not int:
            raise TypeError("Wrong type for group_num,int is needed but actually is " + str(type(group_num)))
        for i in range(group_num):
            group = []
            for j in range(randint(3, 8)):
                group.append(Single(Single.get_random_chromosome()))
            self.__groups.append(group)

    def calculate_fitness(self, sample):
        # if index >= self.__member_num or index < 0:
        #     raise IndexOutOfRangeError("Illegal index")
        # chromosome = self.__member_list[index]

        return np.around(random(0, 1), 2)

    def round_select(self):
        alive_single = None
        return alive_single

    def recombination(self):
        for igroup in range(len(self.__groups)):
            evolution_group = []
            for i in range(len(self.__groups[igroup])):
                evolution_group.append(self.round_select())
            self.__groups[igroup] = evolution_group

    def crossover(self):
        for group in self.__groups:
            index = 0
            while index < len(group):
                # __recombine_digit=(randint(0, Single().get_chromosome_width() / 2),
                #  randint(Single().get_chromosome_width() / 2, Single().get_chromosome_width() - 1))
                for digit in self.__recombine_digit:
                    temp = group[index].chromosome[digit]
                    group[index].chromosome[digit] = group[index + 1].chromosome[digit]
                    group[index + 1].chromosome[digit] = temp
                index += 2

    def mutation(self):
        for group in self.__groups:
            for index in range(len(group)):
                for digit in range(Single().get_chromosome_width()):
                    r = random()
                    if r < self.__mutation_probability:
                        group[index][digit] = 1 - group[index][digit]


if __name__ == "__main__":
    print(Single().get_chromosome_width())
