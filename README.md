Objective:

Train a network which can predict depth and mask of an image given background and foreground images.

Architecture:

![Architecture](https://user-images.githubusercontent.com/19210895/82811808-3e4e3680-9eaf-11ea-83fb-ca259e7f35fa.JPG)

Link to model:
https://github.com/Madhusudhanbandi/EVA-S15/blob/master/model_a15.py

Data Preparation:

Selected a random background and foreground images from repository of 100 and 400k respectively and selected corresponding mask and depth images as target to train our network.

Different data augmentation like Normalize, cutout were used for BG and FG images to avoid overfitting of the model.

![Train data](https://user-images.githubusercontent.com/19210895/82811976-99802900-9eaf-11ea-914a-86ba5d179b84.JPG)


Training:

To train our model tried with different loss functions like MSELoss, BCELoss, BCELossWithLogits, and got good results with BCELoss
There were 10M parameters in our model.

To evaluate model used SSIM (Structural Similarity Index) for both mask and depth images to compare with real mask and depth images.

Results:
Link to notebook

https://github.com/Madhusudhanbandi/EVA-S15/blob/master/EVA_S15_ASSIG.ipynb

![Predicted_MD_images](https://user-images.githubusercontent.com/19210895/82812050-bae11500-9eaf-11ea-97dd-094870129cfc.JPG)


Reference for architecture:
https://link.springer.com/article/10.1007/s11263-019-01183-3

