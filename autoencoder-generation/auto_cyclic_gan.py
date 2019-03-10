import os
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pypianoroll as ppr
from matplotlib import pyplot as plt

class deepautoencoder(nn.Module):
    def __init__(self):
        super(deepautoencoder, self).__init__()

        self.hidsize = 200

        self.enc = nn.Sequential(
            nn.Linear(4*16*84,self.hidsize),
            nn.ReLU(True)
        )

        self.dec = nn.Sequential(
            nn.Linear(self.hidsize,4*16*84),
            nn.Sigmoid()
        )

    def forward(self,x,epoch):
        x = self.enc(x)
        x = self.dec(x)
        return x

#parameter
learning_rate = 1e-4
num_epochs = 2000
batch_size = 1000

#load the data
data_dir = "cyclicGAN_DB/"
data_loc = "pop_train_piano.npy"
data = np.load(data_dir + data_loc)
data = data.reshape(len(data),4*1344).astype(float)
print(data.shape)

X = torch.from_numpy(data).float()

train_data = torch.utils.data.TensorDataset(X,X)
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

model = deepautoencoder().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    total_loss = 0
    k = 0
    for data in dataloader:
        song = Variable(data[0]).cuda()

        output = model(song,epoch)
        loss = criterion(output,song)
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        k += 1
        total_loss += loss.data[0]
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, total_loss/k))
    out = np.reshape(output[0,:].detach().cpu().numpy(),(4*16,84))
    print(np.min(out))
    print(np.max(out))

    out = np.pad(out,((0,0),(0,44)),'constant',constant_values=(0))
    if epoch%10 == 0:
    #if True:
        pianoroll = np.reshape(song[0,:].cpu().numpy(),(4*16,84))
        pianoroll = np.pad(pianoroll,((0,0),(0,44)),'constant',constant_values=(0))
        trackOrg = ppr.Track(pianoroll=pianoroll)
        out = out > 0.5#0.75*np.max(out)
        #trackNew = ppr.Track(pianoroll=np.reshape(output[0,:,:].detach().cpu().numpy(),(4*120,128)))
        trackNew = ppr.Track(pianoroll=out)

        multitrack = ppr.Multitrack(tracks=[trackOrg,trackNew])
        fig, axs = multitrack.plot()
        """
        fig,(ax1,ax2) = plt.subplots(2)
        fig,ax1 = trackOrg.plot()
        fix,ax2 = trackNew.plot()
        """
        plt.savefig("figures\\"+str(epoch)+".png")
        #plt.show()

torch.save(model.state_dict(), "current_pop.pt")
