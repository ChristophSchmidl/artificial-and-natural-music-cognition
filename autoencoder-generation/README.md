# The stimuli files
The simuli files have the following structure:
- song_Xorg.mid the original song
- song_X_candidate_0.mid the generated song of the autoencoder
- song_X_org_auto.mid reconstruction of the orginal song of the autoencoder

So we want to use song_Xorg.mid as the original song file and compare it against song_X_candidate_0.mid.

# Generate your own songs
Please download the dataset from https://goo.gl/ZK8wLW and copy the following files to the folder cyclicGAN_DB:
- classic_train_piano.npy
- jazz_train_piano.npy
- pop_train_piano.npy
