import matplotlib.pyplot as plt
import pandas as pd

f = pd.read_csv('result0028.csv')
x = f.iloc[: , :1]
y = f.loc[: , "score"]
plt.plot(x,y)
plt.title('ShanghaiTech test video 1_028')
plt.xlabel('Frame Numbers')
plt.ylabel("Regularity Score")
plt.savefig("Graphs/result0028.jpg")