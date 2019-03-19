import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('Song 0', 'Song 1', 'Song 2', 'Song 3', 'Song 4', 'Song 5','Song 6', 'Song 7', 'Song 8', 'Song 9', 'Song 10', 'Song 11', 'Song 12', 'Song 13', 'Song 14')
y_pos = np.arange(len(objects))
performance = [0.13,0.24,0.39,0.35,0.15,0.40,0.27,0.15,0.40,0.42,0.31,0.58,0.21,0.21,0.31]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylim([0,1])
chance = plt.axhline(y=0.5,color='r',linestyle='-')
up = plt.axhline(y=0.6,color='g',linestyle='-')
low = plt.axhline(y=0.4,color='g',linestyle='-')

plt.legend((chance,up,low),('Chance level','Upper Sig Threshold','Lower Sig Threshold'))

plt.ylabel('Mean Vote')
plt.title('Jazz Song Examples t-test Means')

plt.show()
