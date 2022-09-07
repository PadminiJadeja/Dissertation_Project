import matplotlib.pyplot as plt
import pandas as pd

f = pd.read_csv('result012.csv')
x = f.iloc[: , :1]
y = f.loc[: , "score"]
plt.plot(x,y)
plt.title('UCSD Ped2 test video 12')
plt.xlabel('Frame Numbers')
plt.ylabel("Regularity Score")
plt.savefig("Graphs/result012.jpg")