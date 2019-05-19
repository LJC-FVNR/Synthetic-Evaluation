from AHP import G1
from entropy import entropy
from Evaluation import fuzzy
import pandas as pd
import numpy as np
pd.set_option('display.width',200)
if __name__ == "__main__":
    # -----------数据引入-------------
    G1_array = pd.read_csv("TJU-LD-G1.csv")
    Entropy_array = pd.read_csv("TJU-LD-Entropy.csv")
    Fuzzy_array1 = pd.read_csv("TJU-LD-Fuzzy1.csv")
    Fuzzy_array2 = pd.read_csv("TJU-LD-Fuzzy2.csv")
    Fuzzy_array3 = pd.read_csv("TJU-LD-Fuzzy3.csv")
    Fuzzy_array4 = pd.read_csv("TJU-LD-Fuzzy4.csv")
    Fuzzy_array5 = pd.read_csv("TJU-LD-Fuzzy5.csv")
    Fuzzy_array6 = pd.read_csv("TJU-LD-Fuzzy6.csv")

    # -----------数据处理-------------
    # -------------G1----------------
    G1_label = list(G1_array.columns)
    G1_sort1 = np.array(G1_array.iloc[0])
    G1_iptc1 = np.array(G1_array.iloc[1].dropna(how='all'))
    G1_sort2 = np.array(G1_array.iloc[2])
    G1_iptc2 = np.array(G1_array.iloc[3].dropna(how='all'))
    G1_sort3 = np.array(G1_array.iloc[4])
    G1_iptc3 = np.array(G1_array.iloc[5].dropna(how='all'))
    G1_sort4 = np.array(G1_array.iloc[6])
    G1_iptc4 = np.array(G1_array.iloc[7].dropna(how='all'))
    G1_sort5 = np.array(G1_array.iloc[8])
    G1_iptc5 = np.array(G1_array.iloc[9].dropna(how='all'))
    G1_sort6 = np.array(G1_array.iloc[10])
    G1_iptc6 = np.array(G1_array.iloc[11].dropna(how='all'))

    # ------------Entropy-------------
    ent = np.array(Entropy_array).T

    # ------------Fuzzy---------------
    F_arr1 = np.array(Fuzzy_array1.iloc[0:12])
    F_scr1 = np.array(Fuzzy_array1.iloc[12])
    F_arr2 = np.array(Fuzzy_array2.iloc[0:12])
    F_scr2 = np.array(Fuzzy_array2.iloc[12])
    F_arr3 = np.array(Fuzzy_array3.iloc[0:12])
    F_scr3 = np.array(Fuzzy_array3.iloc[12])
    F_arr4 = np.array(Fuzzy_array4.iloc[0:12])
    F_scr4 = np.array(Fuzzy_array4.iloc[12])
    F_arr5 = np.array(Fuzzy_array5.iloc[0:12])
    F_scr5 = np.array(Fuzzy_array5.iloc[12])
    F_arr6 = np.array(Fuzzy_array6.iloc[0:12])
    F_scr6 = np.array(Fuzzy_array6.iloc[12])

    # ------------计算G1---------------
    subj_weight = []
    for i in range(1,7):
        print("G1法——专家{}".format(i))
        gg = G1(eval('G1_iptc'+str(i)), ['Ip'+str(j) for j in range(1, 13)], eval('G1_sort'+str(i)))
        gg.construct().describe()
        subj_weight.append(gg.weight)
    subj_weight = np.array(subj_weight)
    subject = np.mean(subj_weight, axis=0)
    # ------------计算Entropy---------------
    e, object = entropy(ent)

    # ------------综合权重---------------
    w_fin = 0.6*subject+0.4*object
    print("综合权重为：")
    print(w_fin)

    # ------------综合评价---------------
    cons = []
    for i in range(1,7):
        fuzz = fuzzy(eval('F_arr'+str(i)), w_fin, eval('F_scr'+str(i)), name="专家"+str(i))
        fuzz.construct().describe()
        cons.append(fuzz.consequence)
    fin_fuzzy = np.array(cons).mean()
    fin_init = (ent.mean(axis=1)*w_fin).sum()
    f = 0.6*fin_fuzzy+0.4*fin_init
    print("初步评价：{}".format(fin_init))
    print("模糊综合评价：{}".format(fin_fuzzy))
    print("总评价：{}".format(f))
