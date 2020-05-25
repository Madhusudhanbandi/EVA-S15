Objective:

Train a network which can predict depth and mask of an image given background and foreground images.

Architecture:

Link to model:

Data Preparation:

Selected a random background and foreground images from repository of 100 and 400k respectively and selected corresponding mask and depth images as target to train our network.

Different data augmentation like Normalize, cutout were used for BG and FG images to avoid overfitting of the model.

Training:

To train our model tried with different loss functions like MSELoss, BCELoss, BCELossWithLogits, and got good results with BCELoss
There were 10M parameters in our model.

To evaluate model used SSIM (Structural Similarity Index) for both mask and depth images to compare with real mask and depth images.

Results:

Reference for architecture:
https://link.springer.com/article/10.1007/s11263-019-01183-3

