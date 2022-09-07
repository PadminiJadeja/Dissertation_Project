import matplotlib.pyplot as plt
import pandas as pd

f = pd.read_csv('result002.csv')
x = f.iloc[: , :1]
y = f.loc[: , "score"]
plt.plot(x,y)
plt.title('UCSD Ped1 test video 2')
plt.xlabel('Frame Numbers')
plt.ylabel("Regularity Score")
plt.savefig("Graphs/result002.jpg")