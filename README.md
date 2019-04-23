### About
Helpers package for semantic segmentation tasks using U-Net models implemented in Keras.

### Features:           
- [x] Vanilla U-Net implementation based on [the original paper](https://arxiv.org/pdf/1505.04597.pdf){:target="_blank"}
- [x] Customizable U-Net:
    - [x] batch norm or/and dropout
    - [x] number of starting filters
    - [x] number of "unet" conv layers
- [x] U-Net optimized for satellite images based on [DeepSense.AI Kaggle competition entry](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/){:target="_blank"}
- [x] Utils:
    - [x] Plotting images and masks
    - [x] Plotting images masks and predictions with overlay (prediction on top of original image)
    - [x] Plotting training history
    - [x] Data augmentation helper function
- [x] Notebooks (examples):
    - [x] Training custom U-Net for whale tails segmentation example
    - [ ] Semantic segmentation for satellite images
    - [ ] Semantic segmentation for medical images



