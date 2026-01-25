import pickle
import matplotlib.pyplot as plt
import numpy as np

def makeGraph(x, y, c):
    plt.scatter(x, y, c=c)

file = open('behaviorSpace', 'rb')
data = pickle.load(file)
file.close()

for i, score in enumerate(data['scores']):
    if score>0.07:
        del data['seperability'][i]
        del data['generalizability'][i]
        del data['scores'][i]
        del data['mc'][i]

plt.figure()
plt.scatter(data['mc'], data['scores'])
plt.xlabel("MC")
plt.ylabel("Error")

plt.figure()
plt.scatter(data['generalizability'], data['scores'])
plt.xlabel("KR")
plt.ylabel("Error")

plt.show()

# makeGraph(data["mc"], data["scores"])
# makeGraph(data["generalizability"], data["scores"])
# makeGraph(data["scores"], np.array(data['generalizability'])-np.array(data['seperability']), data["scores"])
# makeGraph(data["seperability"], data["generalizability"], data["scores"])