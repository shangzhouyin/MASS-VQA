import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

path = '../logs/vizwiz/2022-11-15_12:13:58/final_log.pth'
results = torch.load(path)

val_acc = torch.FloatTensor(results['tracker']['val_acc'])
val_acc = val_acc.mean(dim=1).numpy()



val_acc=np.insert(val_acc,0,0)

train_acc = torch.FloatTensor(results['tracker']['train_acc'])
train_acc = train_acc.mean(dim=1).numpy()


train_acc=np.insert(train_acc,0,0)

plt.gca().set_color_cycle(['blue', 'red'])


axes = plt.gca()
axes.set_ylim([0,1])

plt.plot(val_acc)
plt.plot(train_acc)

fig_acc = plt.gcf()
plt.show()
#plt.savefig('val_acc.png')
plt.figure()
fig_acc.savefig('train_val_5ans_50epochs.png', dpi = 1000)

