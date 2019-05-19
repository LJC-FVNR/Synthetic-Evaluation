import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width',200)
class TOPSIS:
    # 输入：归一化评价矩阵，权重 / 输出：全信息最优解、最劣解；相对贴进度
    def __init__(self, array, weight, label_index=[], label_objects=[]):
        if label_index == []:
            label_index = ['I' + str(i) for i in range(weight.__len__())]
        if label_objects == []:
            label_objects = ['O' + str(i) for i in range(array.__len__())]
        self.array = np.array(array)
        if self.array.shape[1] != weight.__len__():
            print('请检查原始数据')
        self.weight = weight
        self.label_index = label_index
        self.label_objects = label_objects
        self.max = []
        self.min = []
        # 加权矩阵
        self.weighted_array = self.array * np.array(self.weight)
        self.weighted_max = []
        self.weighted_min = []
        # 竖向数据
        self.d_worst = []
        self.d_best = []
        self.relative_closeness = []

    def construct(self):
        for i in range(self.array.shape[1]):
            self.max.append(self.array[:, i].max())
            self.min.append(self.array[:, i].min())
        self.weighted_max = np.array(self.max) * np.array(self.weight)
        self.weighted_min = np.array(self.min) * np.array(self.weight)
        for i in range(self.array.shape[0]):
            self.d_best.append(np.linalg.norm(self.weighted_array[i] - self.weighted_max))
            self.d_worst.append(np.linalg.norm(self.weighted_array[i] - self.weighted_min))
            self.relative_closeness.append(self.d_worst[i] / (self.d_worst[i] + self.d_best[i]))
        return self

    def describe(self):
        objects = self.label_objects.extend(['best', 'worst'])
        index = self.label_index.extend(['d-worst', 'd-best', 'RC'])
        array = self.array
        array = np.column_stack((array, self.d_worst))
        array = np.column_stack((array, self.d_best))
        array = np.column_stack((array, self.relative_closeness))
        max = np.array(self.max)
        min = np.array(self.min)
        max = np.append(max, [None, None, None])
        min = np.append(min, [None, None, None])
        array = np.row_stack((array, max))
        array = np.row_stack((array, min))
        des = pd.DataFrame(array, index=self.label_objects, columns=self.label_index)
        print(des)


class fuzzy:
    # 输入：评价矩阵、指标权重、评语集打分、指标名称 / 输出：得分
    def __init__(self, array, weight, score, name="U"):
        self.array = np.array(array)
        self.weight = np.array(weight)
        self.score = np.array(score)
        self.remark = ['v' + str(i+1) for i in range(self.array.shape[1])]
        self.label = ['U' + str(i+1) for i in range(self.array.shape[0])]
        self.name = name

        self.fz = None
        self.consequence = None

    def construct(self):
        o = []
        for i, j in zip(self.array, self.weight):
            temp = i
            for value, pos in zip(i, range(i.__len__())):
                temp[pos] = min(value, j)
            o.append(temp)
        o = np.array(o)
        fz = o.max(axis=0)
        fz_nm = fz/fz.sum()
        self.fz = fz_nm
        self.consequence = (fz_nm * self.score).sum()
        return self

    def describe(self):
        print("模糊综合评价——{}: ".format(self.name))
        des1 = [str(i)+"("+str(j)+")" for i, j in zip(self.remark, self.score)]
        des1.append("最终得分   ")
        des2 = list(self.fz)
        des2.append(self.consequence)
        des = pd.DataFrame([des1,des2],index=["评语","评价"])
        print(des)

if __name__ == "__main__":
    """
    a = [[60,100,94.45412,73.3714460,60,94.45412,90.25884],[82.96124,73.09076,100,60,100,100,60],[100,60,60,100,77.70028,60,100],[89.47036,62.5968,74.60848,64.73028,90.25884,87.30424,77.70028]]
    w = [0.1377174,0.2794902,0.183881,0.1116034,0.1103296,0.0873558,0.0896218]
    t = TOPSIS(a,w)
    t.construct().describe()
    """
    array = [[0.3, 0.3, 0.35, 0.05, 0],
             [0.2, 0.2, 0.4, 0.16, 0.04],
             [0.3, 0.2, 0.15, 0.05, 0.1],
             [0.9, 0.02, 0.04, 0.02, 0.02]]
    weight = [0.199, 0.22, 0.345, 0.237]
    score = [90,80,65,45,30]
    U1 = fuzzy(array,weight,score)
    U1.construct().describe()
