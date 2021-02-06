import numpy as np
import random
from PIL import Image, ImageOps, ImageEnhance
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
import cv2
import torch
from skimage.transform import AffineTransform, warp

"""
From https://www.kaggle.com/corochann/deep-learning-cnn-with-chainer-lb-0-99700
"""


def affine_image(img):
    """

    Args:
        img: (h, w) or (1, h, w)

    Returns:
        img: (h, w)
    """
    # ch, h, w = img.shape
    # img = img / 255.
    if img.ndim == 3:
        img = img[0]

    # --- scale ---
    min_scale = 0.8
    max_scale = 1.2
    sx = np.random.uniform(min_scale, max_scale)
    sy = np.random.uniform(min_scale, max_scale)

    # --- rotation ---
    max_rot_angle = 7
    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    # --- shear ---
    max_shear_angle = 10
    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.

    # --- translation ---
    max_translation = 4
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)

    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                            translation=(tx, ty))
    transformed_image = warp(img, tform)
    assert transformed_image.ndim == 2
    return transformed_image


def crop_char_image(image, threshold=5. / 255.):
    assert image.ndim == 2
    is_black = image > threshold

    is_black_vertical = np.sum(is_black, axis=0) > 0
    is_black_horizontal = np.sum(is_black, axis=1) > 0
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    return cropped_image


def resize(image, size=(128, 128)):
    return cv2.resize(image, size)


"""
From https://www.kaggle.com/haqishen/augmix-based-on-albumentations
"""


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


# solarize will have bug
augmentations = [
    autocontrast, shear_x, shear_y,
    translate_x, translate_y, equalize, posterize, solarize
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    return image - 127


def apply_op(image, op, severity):
    image = np.clip(image, 0, 255)
    pil_img = Image.fromarray(image, mode="RGB")  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img).astype(np.float32)


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    image = np.float32(image)
    ws = np.float32(
        np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug
    #         mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * image + m * mix
    #     mixed = (1 - m) * normalize(image) + m * mix
    return mixed


class RandomAugMix(ImageOnlyTransform):

    def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, image, **params):
        image = augment_and_mix(
            image,
            self.severity,
            self.width,
            self.depth,
            self.alpha
        )
        return image


class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                        int(i * grid_h): int(i * grid_h + grid_h / 2),
                        int(j * grid_w): int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                            int(i * grid_h + grid_h / 2): int(i * grid_h + grid_h),
                            int(j * grid_w + grid_w / 2): int(j * grid_w + grid_w)
                            ] = self.fill_value

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:, :, np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h + h, rand_w:rand_w + w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


"""
From https://www.kaggle.com/haqishen/augmix-based-on-albumentations
"""


def add_gaussian_noise(x, sigma):
    x += np.random.randn(*x.shape) * sigma
    x = np.clip(x, 0., 1.)
    return x


def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio


def apply_aug(aug, image):
    return aug(image=image)['image']


############################################################ self define class
class CropCharImage(ImageOnlyTransform):

    def __init__(self, threshold=20, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.threshold = threshold

    def apply(self, image, **params):
        image = crop_char_image(image, threshold=self.threshold)
        return image


############################################################ from https://www.kaggle.com/hengck23
# helper --
def make_grid_image(width, height, grid_size=16):
    image = np.zeros((height, width), np.float32)
    for y in range(0, height, 2 * grid_size):
        for x in range(0, width, 2 * grid_size):
            image[y: y + grid_size, x:x + grid_size] = 1

    # for y in range(height+grid_size,2*grid_size):
    #     for x in range(width+grid_size,2*grid_size):
    #          image[y: y+grid_size,x:x+grid_size] = 1

    return image


# ---

def do_identity(image, magnitude=None):
    return image


# *** geometric ***

def do_random_projective(image, magnitude=0.5, p=0.5):
    mag = np.random.uniform(-1, 1) * 0.5 * magnitude

    height, width = image.shape[:2]
    x0, y0 = 0, 0
    x1, y1 = 1, 0
    x2, y2 = 1, 1
    x3, y3 = 0, 1

    mode = np.random.choice(['top', 'bottom', 'left', 'right'])
    if mode == 'top':
        x0 += mag;
        x1 -= mag
    if mode == 'bottom':
        x3 += mag;
        x2 -= mag
    if mode == 'left':
        y0 += mag;
        y3 -= mag
    if mode == 'right':
        y1 += mag;
        y2 -= mag

    s = np.array([[0, 0], [1, 0], [1, 1], [0, 1], ]) * [[width, height]]
    d = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3], ]) * [[width, height]]
    transform = cv2.getPerspectiveTransform(s.astype(np.float32), d.astype(np.float32))

    image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


def do_random_perspective(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        mag = np.random.uniform(-1, 1, (4, 2)) * 0.25 * magnitude

        height, width = image.shape[:2]
        s = np.array([[0, 0], [1, 0], [1, 1], [0, 1], ])
        d = s + mag
        s *= [[width, height]]
        d *= [[width, height]]
        transform = cv2.getPerspectiveTransform(s.astype(np.float32), d.astype(np.float32))

        image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


def do_random_scale(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        s = 1 + np.random.uniform(-1, 1) * magnitude * 0.5

        height, width = image.shape[:2]
        transform = np.array([
            [s, 0, 0],
            [0, s, 0],
        ], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_shear_x(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        sx = np.random.uniform(-1, 1) * magnitude

        height, width = image.shape[:2]
        transform = np.array([
            [1, sx, 0],
            [0, 1, 0],
        ], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_shear_y(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        sy = np.random.uniform(-1, 1) * magnitude

        height, width = image.shape[:2]
        transform = np.array([
            [1, 0, 0],
            [sy, 1, 0],
        ], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_stretch_x(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        sx = 1 + np.random.uniform(-1, 1) * magnitude

        height, width = image.shape[:2]
        transform = np.array([
            [sx, 0, 0],
            [0, 1, 0],
        ], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_stretch_y(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        sy = 1 + np.random.uniform(-1, 1) * magnitude

        height, width = image.shape[:2]
        transform = np.array([
            [1, 0, 0],
            [0, sy, 0],
        ], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_rotate(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        angle = 1 + np.random.uniform(-1, 1) * 30 * magnitude

        height, width = image.shape[:2]
        cx, cy = width // 2, height // 2

        transform = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


# ----
def do_random_grid_distortion(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        num_step = 5
        distort = magnitude

        # http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
        distort_x = [1 + random.uniform(-distort, distort) for i in range(num_step + 1)]
        distort_y = [1 + random.uniform(-distort, distort) for i in range(num_step + 1)]

        # ---
        height, width = image.shape[:2]
        xx = np.zeros(width, np.float32)
        step_x = width // num_step

        prev = 0
        for i, x in enumerate(range(0, width, step_x)):
            start = x
            end = x + step_x
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + step_x * distort_x[i]

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        yy = np.zeros(height, np.float32)
        step_y = height // num_step
        prev = 0
        for idx, y in enumerate(range(0, height, step_y)):
            start = y
            end = y + step_y
            if end > height:
                end = height
                cur = height
            else:
                cur = prev + step_y * distort_y[idx]

            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        map_x, map_y = np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


# https://github.com/albumentations-team/albumentations/blob/8b58a3dbd2f35558b3790a1dbff6b42b98e89ea5/albumentations/augmentations/transforms.py

# https://ciechanow.ski/mesh-transforms/
# https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
# http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
def do_random_custom_distortion1(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        distort = magnitude * 0.3

        height, width = image.shape
        s_x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        s_y = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        d_x = s_x.copy()
        d_y = s_y.copy()
        d_x[[1, 4, 7]] += np.random.uniform(-distort, distort, 3)
        d_y[[3, 4, 5]] += np.random.uniform(-distort, distort, 3)

        s_x = (s_x * width)
        s_y = (s_y * height)
        d_x = (d_x * width)
        d_y = (d_y * height)

        # ---
        distort = np.zeros((height, width), np.float32)
        for index in ([4, 1, 3], [4, 1, 5], [4, 7, 3], [4, 7, 5]):
            point = np.stack([s_x[index], s_y[index]]).T
            qoint = np.stack([d_x[index], d_y[index]]).T

            src = np.array(point, np.float32)
            dst = np.array(qoint, np.float32)
            mat = cv2.getAffineTransform(src, dst)

            point = np.round(point).astype(np.int32)
            x0 = np.min(point[:, 0])
            x1 = np.max(point[:, 0])
            y0 = np.min(point[:, 1])
            y1 = np.max(point[:, 1])
            mask = np.zeros((height, width), np.float32)
            mask[y0:y1, x0:x1] = 1

            mask = mask * image
            warp = cv2.warpAffine(mask, mat, (width, height), borderMode=cv2.BORDER_REPLICATE)
            distort = np.maximum(distort, warp)
            # distort = distort+warp
        image = distort
    return image


# *** intensity ***
def do_random_contast(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        alpha = 1 + random.uniform(-1, 1) * magnitude
        image = image.astype(np.float32) * alpha
        image = np.clip(image, 0, 1)
    return image


def do_random_block_fade(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        size = [0.1, magnitude]

        height, width = image.shape

        # get bounding box
        m = image.copy()
        cv2.rectangle(m, (0, 0), (height, width), 1, 5)
        m = image < 0.5
        if m.sum() == 0: return image

        m = np.where(m)
        y0, y1, x0, x1 = np.min(m[0]), np.max(m[0]), np.min(m[1]), np.max(m[1])
        w = x1 - x0
        h = y1 - y0
        if w * h < 10: return image

        ew, eh = np.random.uniform(*size, 2)
        ew = int(ew * w)
        eh = int(eh * h)

        ex = np.random.randint(0, w - ew) + x0
        ey = np.random.randint(0, h - eh) + y0

        image[ey:ey + eh, ex:ex + ew] *= np.random.uniform(0.1, 0.5)  # 1 #
        image = np.clip(image, 0, 1)
    return image


# *** noise ***
# https://www.kaggle.com/ren4yu/bengali-morphological-ops-as-image-augmentation
def do_random_erode(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        s = int(round(1 + np.random.uniform(0, 1) * magnitude * 6))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s, s)))
        image = cv2.erode(image, kernel, iterations=1)
    return image


def do_random_dilate(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        s = int(round(1 + np.random.uniform(0, 1) * magnitude * 6))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s, s)))
        image = cv2.dilate(image, kernel, iterations=1)
    return image


def do_random_sprinkle(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        size = 16
        num_sprinkle = int(round(1 + np.random.randint(10) * magnitude))

        height, width = image.shape
        image = image.copy()
        image_small = cv2.resize(image, dsize=None, fx=0.25, fy=0.25)
        m = np.where(image_small > 0.25)
        num = len(m[0])
        if num == 0: return image

        s = size // 2
        i = np.random.choice(num, num_sprinkle)
        for y, x in zip(m[0][i], m[1][i]):
            y = y * 4 + 2
            x = x * 4 + 2
            image[y - s:y + s, x - s:x + s] = 0  # 0.5 #1 #
    return image


# https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
def do_random_noise(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        height, width = image.shape
        noise = np.random.uniform(-1, 1, (height, width)) * magnitude * 0.7
        image = image + noise
        image = np.clip(image, 0, 1)
    return image


def do_random_line(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        num_lines = int(round(1 + np.random.randint(8) * magnitude))

        # ---
        height, width = image.shape
        image = image.copy()

        def line0():
            return (0, 0), (width - 1, 0)

        def line1():
            return (0, height - 1), (width - 1, height - 1)

        def line2():
            return (0, 0), (0, height - 1)

        def line3():
            return (width - 1, 0), (width - 1, height - 1)

        def line4():
            x0, x1 = np.random.choice(width, 2)
            return (x0, 0), (x1, height - 1)

        def line5():
            y0, y1 = np.random.choice(height, 2)
            return (0, y0), (width - 1, y1)

        for i in range(num_lines):
            p = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4, 1, 1])
            func = np.random.choice([line0, line1, line2, line3, line4, line5], p=p / p.sum())
            (x0, y0), (x1, y1) = func()

            color = np.random.uniform(0, 1)
            thickness = np.random.randint(1, 5)
            line_type = np.random.choice([cv2.LINE_AA, cv2.LINE_4, cv2.LINE_8])

            cv2.line(image, (x0, y0), (x1, y1), color, thickness, line_type)

    return image


# batch augmentation that uses pairing, e.g mixup, cutmix, cutout #####################
def make_object_box(image):
    m = image.copy()
    cv2.rectangle(m, (0, 0), (236, 137), 0, 10)
    m = m - np.min(m)
    m = m / np.max(m)
    h = m < 0.5

    row = np.any(h, axis=1)
    col = np.any(h, axis=0)
    y0, y1 = np.where(row)[0][[0, -1]]
    x0, x1 = np.where(col)[0][[0, -1]]

    return [x0, y0], [x1, y1]


def do_random_batch_mixup(input, onehot):
    batch_size = len(input)

    alpha = 0.4  # 0.2  #0.2,0.4
    gamma = np.random.beta(alpha, alpha, batch_size)
    gamma = np.maximum(1 - gamma, gamma)

    # #mixup https://github.com/moskomule/mixup.pytorch/blob/master/main.py
    gamma = torch.from_numpy(gamma).float().to(input.device)
    perm = torch.randperm(batch_size).to(input.device)
    perm_input = input[perm]
    perm_onehot = [t[perm] for t in onehot]

    gamma = gamma.view(batch_size, 1, 1, 1)
    mix_input = gamma * input + (1 - gamma) * perm_input
    gamma = gamma.view(batch_size, 1)
    mix_onehot = [gamma * t + (1 - gamma) * perm_t for t, perm_t in zip(onehot, perm_onehot)]

    return mix_input, mix_onehot, (perm_input, perm_onehot)


def do_random_batch_cutout(input, onehot):
    batch_size, C, H, W = input.shape

    mask = np.ones((batch_size, C, H, W), np.float32)
    for b in range(batch_size):
        length = int(np.random.uniform(0.1, 0.5) * min(H, W))
        y = np.random.randint(H)
        x = np.random.randint(W)

        y0 = np.clip(y - length // 2, 0, H)
        y1 = np.clip(y + length // 2, 0, H)
        x0 = np.clip(x - length // 2, 0, W)
        x1 = np.clip(x + length // 2, 0, W)
        mask[b, :, y0: y1, x0: x1] = 0
    mask = torch.from_numpy(mask).to(input.device)

    input = input * mask
    return input, onehot, None
