# GMM

if Running Python 2.7, change "range" to "xrange."

You need to change the file name at the bottom to run the code on your image.

This code clusters an image based on colors. Each pixel correspondes to a RGB vector (3 channels) and is treated as independent samples.

A Gaussian Mixture Model is trained using EM update equations given in the following paper:

http://melodi.ee.washington.edu/people/bilmes/mypapers/em.pdf

The model is a mixture of 3 Gaussian distributions. Mu and Sigma are randomly initialized. For each pixel, cluster labels are determined based on likelihood the pixel comes from each distribution, and choosing the maximum. Then a new image is reconstructed such that each pixels correspondes to the cluster mean (Mu).
