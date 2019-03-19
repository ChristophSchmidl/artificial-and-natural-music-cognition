import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('Song 0', 'Song 1', 'Song 2', 'Song 3', 'Song 5','Song 6', 'Song 7', 'Song 8', 'Song 9', 'Song 10', 'Song 11', 'Song 12', 'Song 13', 'Song 14')
y_pos = np.arange(len(objects))
performance = [0.06,0.23,0.11,0.34,0.35,0.15,0.44,0.45,0.16,0.10,0.16,0.06,0.10,0.27]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylim([0,1])
chance = plt.axhline(y=0.5,color='r',linestyle='-')
up = plt.axhline(y=0.6,color='g',linestyle='-')
low = plt.axhline(y=0.4,color='g',linestyle='-')

plt.legend((chance,up,low),('Chance level','Upper Sig Threshold','Lower Sig Threshold'))

plt.ylabel('Mean Vote')
plt.title('Classical Song Examples t-test Means')
plt.show()
