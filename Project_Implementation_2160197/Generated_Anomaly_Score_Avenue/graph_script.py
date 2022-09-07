import matplotlib.pyplot as plt
import pandas as pd

f = pd.read_csv('result005.csv')
x = f.iloc[: , :1]
y = f.loc[: , "score"]
plt.plot(x,y)
plt.title('Avenue test video 5')
plt.xlabel('Frame Numbers')
plt.ylabel("Regularity Score")
plt.savefig("Graphs/result005.jpg")