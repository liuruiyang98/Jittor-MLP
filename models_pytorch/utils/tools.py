from typing import List


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

def check_sizes(image_size, patch_size):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)
    assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
    num_patches = (image_height // patch_height) * (image_width // patch_width)
    return num_patches