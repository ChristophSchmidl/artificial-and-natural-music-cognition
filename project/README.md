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