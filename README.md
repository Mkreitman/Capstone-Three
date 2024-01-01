
# Mushroom Edibility Classifier

## Overview
This project focuses on building a machine learning model to classify the edibility of mushrooms based on images. By utilizing pre-trained models from the Fastai library, the developed classifier achieved a 79% precision score and an 86% recall score for poisonous mushrooms. While the model shows promise, further refinement is necessary before public deployment.

## Data Processing

### Data Collection and Quality Check
The dataset, obtained from Kaggle.com, consisted of images categorized into edible and poisonous mushroom classes. Data wrangling involved renaming and consolidating images based on their toxicity class. A thorough check using the Python Image Library ensured the integrity of all 3,400 image.  Just under 2/3 of the images were of poisonous-labeled mushrooms.

![Image_class_distribution](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/Mushroom%20Image%20Class%20Distribution.png)

### Image Anomalies Handling
Identified issues like watermarks and URL banners in some images. Applied inpainting techniques to address watermark anomalies, although considering original images for further augmentation post-model evaluation.

![Inpaint_biharomonic](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/Inpaint_biharmonic.png)
![denoise](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/denoise.png)

### Exploratory Data Analysis
Explored pixel intensities across RGB channels to understand lighting conditions and potential biases. Noted a prevalence of red and green hues, with blue hues less prominent.

![RBG_review](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/RBG_pixel_intensity.png)

## Model Training and Evaluation

### Data Preparation
Converted images to .jpg format and split them into training, validation, and testing sets. Constructed Dataloaders using Fastai’s Datablock API.

### Model Selection
Trained and evaluated three models using Fastai’s vision_learner: 

1) Resnet34 with 128-pixel images (4 epochs)
![Model_1_epoch_results]([https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/model_1_cm.png](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/Model_1.png))

2) Resnet34 with 224-pixel images (4 epochs)
![Model_2_epoch_results](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/Model_2.png)

3) Resnet50 with 128-pixel images (4 epochs)
![Model_3_epoch_results]([https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/model_1_cm.png](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/Model_3.png))

### Performance Analysis
Model 2 outperformed others, exhibiting superior validation loss, precision, and recall. Confusion matrices and precision-recall analysis were employed for detailed performance evaluation.

![Model_1](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/model_1_cm.png)

![Model_2](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/model_2_cm.png)

![Model_3](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/model_3_cm.png)

![Poisonous_precision_and_recall_comparison](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/Poisonous_test_precision_and_recall_results.png)

Demonstrated below, Model 2 had the closest training and validation loss measures comparatively to the other two which indicate low variance and suggests that it is not overfitting or underfitting the training data significantly.

![Comparison_of_training_and_validation_loss](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/Variance_check.png)

## Conclusions
For precision in predicting poisonous mushrooms, Model 1 utilizing Resnet50 with 128-pixel images achieved 81% accuracy. For maximizing recall, the Model 2 utilizing Resnet34 with 224-pixel images excelled with an 86% accuracy. I decided that recall was imperative to selecting a high quality model from a health safety standpoint, so Model 2 is my selection for this project.

Here are some examples of Model 2's predictions:

![Model_2_prediction_results](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/figures/model_2_predictions.png)

## Future Work
Improving model performance by adding high-quality images focusing on top and bottom of the mushroom's cap and stem with accurate edibility label. Exploring advanced hardware, such as a GPU system, for processing larger image sizes. Addressing watermark issues through techniques like thresholding and edge detection, ensuring crucial image features are preserved. Adjusting the blue pixel distribution to align more closely with the red and green channels' distribution curves.

## Project Writeup:
[Final Project PDF](https://github.com/Mkreitman/Capstone-Three/blob/main/reports/Capstone_Final_Report.pdf)
