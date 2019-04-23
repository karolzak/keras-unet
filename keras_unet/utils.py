import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# Runtime data augmentation
def get_augmented(
    X_train, 
    Y_train, 
    X_val=None,
    Y_val=None,
    batch_size=32, 
    seed=0, 
    data_gen_args = dict(
        rotation_range=10.,
        #width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=5,
        #zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='constant'
    )):


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)
    
    train_generator = zip(X_train_augmented, Y_train_augmented)

    if not (X_val is None) and not (Y_val is None):
        # Validation data, no data augmentation, but we create a generator anyway
        X_datagen_val = ImageDataGenerator(**data_gen_args)
        Y_datagen_val = ImageDataGenerator(**data_gen_args)
        X_datagen_val.fit(X_val, augment=True, seed=seed)
        Y_datagen_val.fit(Y_val, augment=True, seed=seed)
        X_val_augmented = X_datagen_val.flow(X_val, batch_size=batch_size, shuffle=True, seed=seed)
        Y_val_augmented = Y_datagen_val.flow(Y_val, batch_size=batch_size, shuffle=True, seed=seed)

        # combine generators into one which yields image and masks
        val_generator = zip(X_val_augmented, Y_val_augmented)
        
        return train_generator, val_generator
    else:
        return train_generator


def plot_segm_history(history, metrics=['iou', 'val_iou'], losses=['loss', 'val_loss']):
    # summarize history for iou
    plt.figure(figsize=(12,6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle('metrics over epochs', fontsize=20)
    plt.ylabel('metric', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(metrics, loc='center right', fontsize=15)
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(12,6))    
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle('loss over epochs', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(losses, loc='center right', fontsize=15)
    plt.show()


def mask_to_red(mask, img_size=1024):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    c1 = mask.reshape(img_size,img_size)
    c2 = np.zeros((img_size,img_size))
    c3 = np.zeros((img_size,img_size))
    c4 = mask.reshape(img_size,img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


def mask_to_rgba(mask, img_size=1024, color='red'):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    zeros = np.zeros((img_size,img_size))
    ones = mask.reshape(img_size,img_size)
    if color == 'red':
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == 'green':
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == 'blue':
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == 'yellow':
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == 'magenta':
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == 'cyan':
        return np.stack((zeros, ones, ones, ones), axis=-1)

    
def plot_imgs(org_imgs, 
              mask_imgs, 
              pred_imgs=None, 
              nm_img_to_plot=10, 
              figsize=4, 
              img_size=1024, 
              alpha=0.5
             ):
    '''
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    '''
    #nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    if  not (pred_imgs is None):
        cols = 4
    else:
        cols = 3
        
    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols*figsize, nm_img_to_plot*figsize))
    axes[0, 0].set_title("original", fontsize=15) 
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15) 
        axes[0, 3].set_title("overlay", fontsize=15) 
    else:
        axes[0, 2].set_title("overlay", fontsize=15) 
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id])
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id].reshape((img_size,img_size)), cmap='gray')
        axes[m, 1].set_axis_off()        
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id].reshape((img_size,img_size)), cmap='gray')
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id])
            axes[m, 3].imshow(mask_to_red(pred_imgs[im_id].reshape((img_size,img_size)), img_size=img_size), alpha=alpha)
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id])
            axes[m, 2].imshow(mask_to_red(mask_imgs[im_id].reshape((img_size,img_size)), img_size=img_size), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1

    plt.show()