import numpy as np
array_loaded = np.load("/dcs/pg21/u2160197/Desktop/Diss_Project/ShanghaiTech/testing/test_frame_mask/01_0051.npy")
# from itertools import islice


x = array_loaded
print(x)

# split_points = [1439,1211,923,947,1007,1283,605,36,1175,841,472,1271,549,507,1001,740,426,294,248,273,76]
# parts = [my_list[ind:ind + k] for ind in split_points]
# Inputt = iter(my_list)
# Output = [list(islice(Inputt, elem))
#         for elem in split_points]
# print(Output[19])
f = open("Test01/frames","a")

# for x in Output:
out=[]
count=0
print(x)
for i in range(1,len(x)):

    prev=x[i-1]
    if x[i]==prev:
        count+=1
    elif count and x[i]!=prev:
        if x[i] ==0:
            out.append([i-count-1+1,i-1+1])
        count=0
if count:
    if x[i]==1:
        out.append([i-count+1,i+1])
    print(out)
    f.write(str(out))
    f.write("\n")
    # break
f.close()