import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.cluster.hierarchy import dendrogram, average

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
class ahp:

    def __init__(self, u):
        u = pd.DataFrame(u)
        self.row = u.shape[0]  # 行数
        self.col = u.shape[1]  # 列数
        self.nev, self.lmd, self.CR, self.consistency = 0, 0, 0, 0
        if self.col > self.row:
            self.col = self.col-1
            self.udata = u.iloc[:, 1:]      # 纯数据
            self.label = np.array(u.iloc[:, 0])
        elif self.col < self.row:
            self.row = self.row-1
            self.udata = u.iloc[1:, :]      # 纯数据
            self.label = np.array(u.iloc[0, :])
        else:
            self.udata = u
            label = []
            for lab in range(0, self.col):
                label.append("U" + repr(lab))
            self.label = np.array(label)
        with open("ri.json", "r") as ri:
            ri.seek(0)
            risheet = json.load(ri)         # R.I.
            self.RI = risheet[repr(self.col)]
        self.ced = False

    def __structure(self, data):
        data = pd.DataFrame(data)
        sum = data.sum(axis=0)
        normalizedU = data / sum            # 归一化矩阵
        eigenVector = normalizedU.sum(1)    # 横向特征向量
        EVsum = eigenVector.sum(0)
        normalizedEV = eigenVector / EVsum  # 特征向量归一化W
        UW = []
        for r in range(0, self.row):
            UW.append(np.array(data.iloc[r]).dot(np.array(normalizedEV)))
        UWW = np.array(UW) / normalizedEV
        lmd = UWW.sum(0) / self.col
        CI = (lmd - self.col) / (self.col - 1)  # Constant Index
        CR = CI / self.RI
        consistency = CR < 0.1
        return normalizedEV, lmd, CR, consistency

    def construct(self):
        self.nev, self.lmd, self.CR, self.consistency = self.__structure(data=self.udata)
        self.nev = list(self.nev)
        self.ced = True
        return self

    def write(self):
        print("指标权重为：")
        for i, j in zip(self.nev, self.label):
            print("{} : {:.2%}".format(j, i))
        print("lmd : {}".format(self.lmd))
        print("一致性比率为：{}\n一致性为：{}".format(self.CR, self.consistency))

    def optimize_find(self, adata, delta):
        if self.ced == False:
            print("请先进行计算")
            return False
        opt = np.array(adata)      # 使用结构
        optmin = np.array(adata)
        u = np.array(adata)
        q = np.array(adata)
        for i in range(0, self.col):
            for j in range(0, self.col):
                if i == j:
                    opt[i][j] = 0
                    continue
                u[i][j] = u[i][j]+delta
                u[j][i] = 1/u[i][j]
                a, b, c, d = self.__structure(data=u)
                c = self.CR - c
                opt[i][j] = c
        u = np.array(adata)
        for i in range(0, self.col):
            for j in range(0, self.col):
                if i == j:
                    optmin[i][j] = -5
                    continue
                u[i][j] = u[i][j]-delta
                u[j][i] = 1/u[i][j]
                e, f, g, h = self.__structure(data=u)
                g = self.CR - g
                optmin[i][j] = g
        positionA = np.where(opt == np.max(opt))
        positionB = np.where(optmin == np.max(optmin))
        pointA = opt[int(positionA[0])][int(positionA[1])]
        pointB = optmin[int(positionB[0])][int(positionB[1])]
        if pointA > pointB:
            position = positionA
            sign = "plus"
        elif pointA < pointB:
            position = positionB
            sign = "minus"
        else:
            position = positionB
            sign = "minus"
        deputy = q[int(position[0])][int(position[1])]
        while deputy <= 1/9 or deputy >= 9:
            opt[int(position[0])][int(position[1])] = -1
            position = np.where(opt == np.max(opt))
            deputy = q[int(position[0])][int(position[1])]
        return opt, position, sign

    def optimize(self, times, delta1):
        uo = np.array(self.udata)
        po = []
        for i in range(0, times):
            opt1, position1, sign1 = self.optimize_find(adata=uo, delta=delta1)
            deput = uo[int(position1[0])][int(position1[1])]
            if sign1 == "plus":
                uo[int(position1[0])][int(position1[1])] = deput + delta1
                uo[int(position1[1])][int(position1[0])] = 1/(deput + delta1)
            if sign1 == "minus":
                uo[int(position1[0])][int(position1[1])] = deput - delta1
                uo[int(position1[1])][int(position1[0])] = 1 / (deput - delta1)
            dec = uo[int(position1[0])][int(position1[1])]
            if dec < 1/9:
                uo[int(position1[0])][int(position1[1])] = 1/9
                uo[int(position1[1])][int(position1[0])] = 9
            if dec > 9:
                uo[int(position1[0])][int(position1[1])] = 9
                uo[int(position1[1])][int(position1[0])] = 1/9
            po.append([int(position1[0]), int(position1[1]), sign1])
            a1, b1, c1, d1 = self.__structure(uo)
            self.CR = c1
            if c1 < 0.1:
                break
        return uo, po, self.CR

class G1:

    def __init__(self, weight_ratio, label, order=[]):
        self.wr = np.array(weight_ratio)
        self.label = label
        self.weight = []
        if order == []:
            self.order = range(1, label.__len__() + 1)
        else:
            self.order = order
    def construct(self):
        l = self.wr.__len__()
        temp_weight = []
        rr = 1
        for i in range(l):
            rr = rr + np.array(self.wr[i:]).prod()
        current = 1/rr
        temp_weight.append(current)
        for i in range(l):
            current = current * self.wr[-(i+1)]
            temp_weight.append(current)
        temp_weight.reverse()
        self.weight = temp_weight
        return self

    def write(self):
        print("指标权重为：")
        for i, j in zip(self.weight, self.label):
            print("{} : {:.2%}".format(j, i))

    def describe(self):
        #  wr = list(self.wr).append(' ')
        des = pd.DataFrame([self.order, self.label, self.wr, self.weight], index=['Order', 'Index', 'Ratio', 'Weight'])
        des = des.sort_values('Order', axis=1)
        self.weight = des.iloc[3]
        print(des)

if __name__ == "__main__":
    print("ahp")
    a = G1([1.2, 1.3, 1.1, 1.2, 1.5, 1.2, 1.5], ['B1', 'A1', 'A2', 'B3', 'B2', 'C1', 'C2'], [2,1,3,4,5,6,7,8])
    a.construct().describe()


