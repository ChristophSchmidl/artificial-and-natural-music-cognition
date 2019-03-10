import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
import pypianoroll as ppr
from sklearn.neighbors import NearestNeighbors


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

    def forward(self,x):
        x = self.enc(x)
        x = self.dec(x)
        return x

    def decode(self,x):
        return self.dec(x)

    def encode(self,x):
        return self.enc(x)

def out_to_array(out):
    out = np.reshape(out.detach().cpu().numpy(),(4*16,84))
    out = np.pad(out,((0,0),(0,44)),'constant',constant_values=(0))
    return out > 0.9*np.max(out)

def write(track,filename):
    track = ppr.Track(pianoroll=track)
    track.transpose(36)
    track = ppr.Multitrack(tracks=[track],beat_resolution=4)
    ppr.write(track,filename)

STEP_SIZE = 0.1
STEPS = np.arange(STEP_SIZE,1,STEP_SIZE) #STEP_SIZE as stort to not duble the initial song
JAZZ = False

print("Generate model")
model = deepautoencoder().cuda()
if JAZZ:
    model.load_state_dict(torch.load("current_jazz.pt"))
else:
    model.load_state_dict(torch.load("current_classic.pt"))

print("Loading data")
#shape = (15,64,84,1)

if JAZZ:
    stimuli = np.load("JazzCandidates.npy")
else:
    stimuli = np.load("ClassicalCandidates.npy")
stimuli = stimuli.reshape(len(stimuli),4*1344).astype(float)

if JAZZ:
    dataset = np.load("cyclicGAN_DB/jazz_train_piano.npy")
else:
    dataset = np.load("cyclicGAN_DB/classic_train_piano.npy")
dataset = dataset.reshape(len(dataset),4*1344).astype(float)

print("Encode the dataset to latent space")
X = torch.from_numpy(dataset).float()
X = Variable(X).cuda()
data_latent = model.encode(X).detach().cpu().numpy()

print("Encode candiadtes")
stimuli_var = torch.from_numpy(stimuli).float()
stimuli_var = Variable(stimuli_var).cuda()
stimuli_latent = model.encode(stimuli_var).detach().cpu().numpy()

print("Fit KNN")
neigh = NearestNeighbors(n_neighbors=30)
neigh.fit(data_latent)
print(data_latent.shape)
print(stimuli_latent.shape)


for i in range(len(stimuli_latent)):
    #[[]], to get shape to (1,200) instead of (200,)
    j_lat = stimuli_latent[[i]]
    #first = distance,                           first = same song
    nearest_song_numbers = neigh.kneighbors(j_lat)[1][0,1:]
    song_number = neigh.kneighbors(j_lat)[1][0,0]
    candiadtes = []
    original_auto = out_to_array(model.decode(Variable(torch.from_numpy(data_latent[song_number]).float()).cuda()))
    original = dataset[song_number]
    original = np.reshape(original,(4*16,84))
    original = np.pad(original,((0,0),(0,44)),'constant',constant_values=(0)) > 0.9*np.amax(original)

    for near_nr in nearest_song_numbers:
        #ensure, that the two samples are not from the same song
        if abs(song_number - near_nr) > 30:
            #print("Song number: " + str(song_number))
            #print("Nearest number: " + str(near_nr))
            morph_from = data_latent[song_number]
            morph_to = data_latent[near_nr]
            for distance in STEPS:
                mix_latent = morph_from*(1-distance) + morph_to*distance
                mix_latent = Variable(torch.from_numpy(mix_latent).float()).cuda()
                candiadtes.append(out_to_array(model.decode(mix_latent)))
            #sound found
            break

    if JAZZ:
        if len(candiadtes) > 0:
            write(original_auto,"stimuli/jazz/song_"+str(i)+"_org_auto")
            write(original,"stimuli/jazz/song_"+str(i)+"org")
            for j in range(len(candiadtes)):
                candiadte = candiadtes[j]
                write(candiadte,"stimuli/jazz/"+"song_"+str(i)+"_candidate_"+str(j))
        else:
            print("No candidates found, increase n_neighbors")
            print("Song Nr: " + str(song_number))
            print("Candidate Nrs: " + str(nearest_song_numbers))
            break

        print("Jazz sample " + str(i) + " candidates generated.")
    else:
        if len(candiadtes) > 0:
            write(original_auto,"stimuli/classic/song_"+str(i)+"_org_auto")
            write(original,"stimuli/classic/song_"+str(i)+"org")
            for j in range(len(candiadtes)):
                candiadte = candiadtes[j]
                write(candiadte,"stimuli/classic/"+"song_"+str(i)+"_candidate_"+str(j))
        else:
            print("No candidates found, increase n_neighbors")
            print("Song Nr: " + str(song_number))
            print("Candidate Nrs: " + str(nearest_song_numbers))
            break
        print("Classic sample " + str(i) + " candidates generated.")
