import os
import re
import tensorflow_datasets as tfds
import tensorflow as tf
import torch
import numpy as  np
tf.config.experimental.set_visible_devices([], "GPU")


def get_datasets(dataset_name, batch_size):
    ds_getter = None
    if dataset_name == "brats":
        ds_getter = get_dataset_brats
    elif dataset_name == "fastmri_knee":
        ds_getter = get_dataset_knee_multicoil
    elif dataset_name == "ms_tia_brain":
        ds_getter = get_dataset_ms_tia_brain
    elif dataset_name == "ms_tia_brain_ch2":
        ds_getter = get_dataset_ms_tia_brain_ch2

    return ds_getter(batch_size)


# def create_files(dir, n):
#     for fn in os.listdir(dir):
#         if fn.endswith('.pt'):
#             os.remove(os.path.join(dir, fn))

#     for i in range(1,n+1):
#         files = ["ema_0.9999_%d.pt", "model%d.pt", "opt%d.pt"]
#         for f in files:
#             with open(os.path.join(dir, f%(i*10000)), "a"):
#                 pass

def keep_last_n_checkpoints(dir, n):
    # Lists to hold each type of checkpoint file
    file_dict={}
    regex = re.compile("[^\d]+(\d+)\.pt")
    for f in os.listdir(dir):
        match = regex.search(f)
        if match:
            file_dict.setdefault(int(match.group(1)), []).append(f)

    sorted_files = sorted(file_dict.items())
    while len(sorted_files) > n:
        files_to_remove = sorted_files.pop(0)
        for f in files_to_remove[1]:
            os.remove(os.path.join(dir, f))




def apply_sensitivity_mask(imgs):
    return imgs * choose_sensMasks(len(imgs))

_sensitivity_masks = None
def choose_sensMasks(batch):
    global _sensitivity_masks
    if _sensitivity_masks is None:
        _sensitivity_pos = [(20,20), (0,160), (20,300), (80,80), (80,240), (160,0), 
                            (160,80), (160,160), (160,240), (160,320),(240,80), 
                            (240,240), (300,20), (320,160), (300,300)]
        _sensitivity_masks = np.stack([create_light_mask((320,320), (_sensitivity_pos[i], 1.2, 270)) 
                                       for i in range(15)])[:,None] # (b, 1, 320, 320)
    indices = np.random.choice(_sensitivity_masks.shape[0], size=batch, replace=True)
    return _sensitivity_masks[indices]

def create_light_mask(image_size, light_source):
    # Create an empty mask
    mask = torch.zeros(image_size)

    # Get the light center and maximum intensity
    light_center, max_intensity, light_radius = light_source

    # Generate coordinates grid
    y_coords, x_coords = torch.meshgrid(torch.arange(image_size[0]), torch.arange(image_size[1]))
    distance_to_center = torch.sqrt((x_coords - light_center[0])**2 + (y_coords - light_center[1])**2)

    # Linear drop in intensity
    mask = max_intensity * torch.where(distance_to_center <= light_radius, (1 - distance_to_center/light_radius), torch.zeros_like(distance_to_center))

    # Clip the mask intensity values to the range [0, 1]
    mask = torch.clamp(mask, 0, 1)
    return mask


def get_dataset_brats(batch_size):
    dataset_name="brats"
    def data_wrapper(ds):
        for d in ds:
            yield torch.as_tensor(d.numpy(), dtype=torch.float32), {}

    def prepare_image(d_dict):
        img = d_dict['image']
        img = tf.cast(img, tf.float32) / 255. # each pixel is [0,1]
        img = tf.transpose(img, (2,0,1)) # b w h c => b c w h
        # img = 2 * img - 1.0 
        return img

    train, info = tfds.load(dataset_name, split='train[:70%]', shuffle_files=True, 
                        as_supervised=False, with_info=True) # 26958
    train = train.map(prepare_image)
    train = train.repeat().shuffle(512)
    train = train.batch(batch_size).prefetch(-1)
    print(info)
    print("[dataset] train number: %d" % (info.splits['train'].num_examples * 0.7))

    val, info= tfds.load(dataset_name, split='train[70%:]', shuffle_files=True, 
                        as_supervised=False, with_info=True) # 1662 = 5543 * 0.3
    val = val.map(prepare_image)
    val = val.repeat().shuffle(512)
    val = val.batch(batch_size, drop_remainder=True).prefetch(-1)
    print("[dataset] validate number: %d" % (info.splits['train'].num_examples * 0.3))
    
    return data_wrapper(train), data_wrapper(val)


def get_dataset_knee_multicoil(batch_size):
    dataset_name="fastmri_knee"
    
    def data_wrapper(ds):
        for d in ds:
            yield torch.as_tensor(apply_sensitivity_mask(d.numpy()), dtype=torch.float32), {}

    def prepare_image(d_dict):
        img = d_dict['image']
        img = tf.cast(img, tf.float32) / 255. # each pixel is [0,1]
        img = tf.transpose(img, (2,0,1)) # b w h c => b c w h
        # img = 2 * img - 1.0 
        return img

    train, info = tfds.load(dataset_name, split='train', shuffle_files=True, 
                        as_supervised=False, with_info=True) # 26958
    train = train.map(prepare_image)
    train = train.repeat().shuffle(512)
    train = train.batch(batch_size).prefetch(-1)
    print(info)
    print("[dataset] train number: %d" % (info.splits['train'].num_examples))

    val, info= tfds.load(dataset_name, split='val', shuffle_files=True, 
                        as_supervised=False, with_info=True) # 1662 = 5543 * 0.3
    val = val.map(prepare_image)
    val = val.repeat().shuffle(512)
    val = val.batch(batch_size, drop_remainder=True).prefetch(-1)
    print("[dataset] validate number: %d" % (info.splits['val'].num_examples))
    
    return data_wrapper(train), data_wrapper(val)

def get_dataset_ms_tia_brain(batch_size):
    dataset_name="ms_tia_brain"
    def data_wrapper(ds):
        for d in ds:
            yield torch.as_tensor(d.numpy(), dtype=torch.float32), {}

    def prepare_image(d_dict):
        img = d_dict['image']
        img = tf.cast(img, tf.float32) / 255. # each pixel is [-1,1]
        img = tf.transpose(img, (2,0,1)) # b w h c => b c w h
        return img

    train, info = tfds.load(dataset_name, split='train', shuffle_files=True, 
                        as_supervised=False, with_info=True) 
    train = train.map(prepare_image)
    train = train.repeat().shuffle(512)
    train = train.batch(batch_size).prefetch(-1)
    print(info)
    print("[dataset] train number: %d" % (info.splits['train'].num_examples))

    val, info= tfds.load(dataset_name, split='val', shuffle_files=True, 
                        as_supervised=False, with_info=True) 
    val = val.map(prepare_image)
    val = val.repeat().shuffle(512)
    val = val.batch(batch_size).prefetch(-1)
    print("[dataset] validate number: %d" % (info.splits['val'].num_examples))
    
    return data_wrapper(train), data_wrapper(val)


def get_dataset_ms_tia_brain_ch2(batch_size):
    dataset_name="ms_tia_brain_ch2"
    def data_wrapper(ds):
        for d in ds:
            yield torch.as_tensor(d.numpy(), dtype=torch.float32), {}

    def prepare_image(d_dict):
        img = d_dict['image']
        img = tf.cast(img, tf.float32) / 255. # each pixel is [-1,1]
        img = tf.transpose(img, (2,0,1)) # b w h c => b c w h
        return img

    train, info = tfds.load(dataset_name, split='train', shuffle_files=True, 
                        as_supervised=False, with_info=True) 
    train = train.map(prepare_image)
    train = train.repeat().shuffle(512)
    train = train.batch(batch_size).prefetch(-1)
    print(info)
    print("[dataset] train number: %d" % (info.splits['train'].num_examples))

    val, info= tfds.load(dataset_name, split='val', shuffle_files=True, 
                        as_supervised=False, with_info=True) 
    val = val.map(prepare_image)
    val = val.repeat().shuffle(512)
    val = val.batch(batch_size).prefetch(-1)
    print("[dataset] validate number: %d" % (info.splits['val'].num_examples))
    
    return data_wrapper(train), data_wrapper(val)