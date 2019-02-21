# Music Generation

Genre-specific music generation using Generative Adversarial Networks (GANs)

1. Topic area: Music Generation

2. Research question: Can Generative Adversarial Networks (GANs) be used to generate genre-specific music?

3. Explanation:

Generative Adversarial Networks (GANs) are systems based on two components, namely a generator and a discriminator. The generator is responsible for generating so called "candidates" which were not drawn from the true data distribution but a latent space. The discriminator evaluates the candidates of the generator and has therefore knowledge about the true data distribution. The main goal of the generator is to increase the error rate of the discriminator which means that the generated candidates become more similar to the true data and therefore harder for the discriminator to actually separate between true and "fake" samples.

In this project, already implemented discriminators will be used which are able to classify musical genres. The main work therefore lies on the generator which can be based on several network types which are heavily used in the music domain, namely:

* Recurrent Neural Networks
* Long short-term memory
* Music Transformer (Sequence model based on self-attention)
	* Paper: https://arxiv.org/abs/1809.04281
	* Blog: https://magenta.tensorflow.org/music-transformer
WaveNet
	* Paper: https://arxiv.org/pdf/1609.03499.pdf
	* Blog: https://deepmind.com/blog/wavenet-generative-model-raw-audio/

Datasets:

* The Million Song Dataset: https://labrosa.ee.columbia.edu/millionsong/
* The MagnaTagATune Dataset: http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
* The GTZAN Genre Collection dataset: http://marsyas.info/downloads/datasets.html
* The Lakh MIDI Dataset: https://colinraffel.com/projects/lmd/
* Imagenet: http://www.image-net.org/challenges/LSVRC/


## Papers

* Deep Music Genre - An approach for automatic music
genre detection and tagging using convolutional neural networks: http://cs231n.stanford.edu/reports/2017/pdfs/22.pdf
* MidiNet - A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation: https://arxiv.org/abs/1703.10847
* An End to End Model for Automatic Music Generation - Combining Deep Raw and Symbolic Audio Networks: http://people.bu.edu/bkulis/pubs/manzelli_mume.pdf
* A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music: https://arxiv.org/abs/1803.05428
* Deep Learning for Music: https://cs224d.stanford.edu/reports/allenh.pdf
* Symbolic Music Genre Transfer with CycleGAN: https://www.tik.ee.ethz.ch/file/2e6c8407bf92ce1e47c0faa7e9a3014d/cyclegan-music-style%20(3).pdf
* WAVENET - A GENERATIVE MODEL FOR RAW AUDIO: https://arxiv.org/pdf/1609.03499.pdf

## Articles

* Music generation with Neural Networks — GAN of the week: https://medium.com/cindicator/music-generation-with-neural-networks-gan-of-the-week-b66d01e28200
* Music Generation Using Deep Learning: https://medium.com/datadriveninvestor/music-generation-using-deep-learning-85010fb982e2
* Making Music - When Simple Probabilities Outperform Deep Learning: https://towardsdatascience.com/making-music-when-simple-probabilities-outperform-deep-learning-75f4ee1b8e69

## Similar projects

* MuseGAN: https://salu133445.github.io/musegan/
* Music Transformer - Generating Music with Long-Term Structure: https://magenta.tensorflow.org/music-transformer
* WaveNet - A Generative Model for Raw Audio: https://deepmind.com/blog/wavenet-generative-model-raw-audio/

## Repositories

* https://github.com/gauravtheP/Music-Generation-Using-Deep-Learning
* https://github.com/ybayle/awesome-deep-learning-music
* https://github.com/jisungk/deepjazz
* https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification
* https://github.com/tensorflow/magenta
* https://github.com/hindupuravinash/the-gan-zoo