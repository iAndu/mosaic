import glob
import time
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import transform
from scipy import misc

# Options

# Path to image to be reproduced
image_path = './data/imaginiTest/ferrari.jpeg'
# Path to collection of images used in mosaic
collection_dir = './data/colectie'
# Mime type of images in collection (must be the same for all)
mime_type = 'png'
# Number of horizontal pieces to be used
horizontal_pieces = 100
# Show or not the collection pieces after being read8
show_collection = True
# How the pieces should be arranged: 'random' or 'grid'
arrange_mode = 'grid'
# Criteria to build the mosaic: 'random' or 'mean_color_distance'
criteria = 'mean_color_distance'
# Till how much percent of the image to place randomly images
treshold = 2
# Use different neighbour pics or not
different_neighbours = False


# Read data from CIFAR-10 batches
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return np.asarray(data[b'data']), np.asarray(data[b'labels'])


# Compute the dimensions for the image to be resize given the number of horizontal pieces
def get_new_size(image_size, horizontal_pieces, patch_size):
    aspect_ratio = image_size[1] / image_size[0]
    new_width = horizontal_pieces * patch_size[1]
    new_height = int(new_width / aspect_ratio)
    new_height = new_height - new_height % patch_size[0]
    return new_height, new_width


# Read image collection used to produce the mosaic
def read_collection(directory, mime_type, is_grayscale=False):
    collection = []
    for filename in glob.glob(directory + '/*.' + mime_type):
        im = Image.open(filename)
        if is_grayscale:
            im = im.convert('L')
        im = np.asarray(im)
        collection.append(im)
    return np.asarray(collection)


# Get a 10x10 preview of the pictures collection
def get_preview(collection):
    rows = []
    for i in range(0, 100, 10):
        rows.append(np.hstack(collection[i:i + 10]))
    return np.vstack(rows)


# Get the mean colors of an image collection, with the possibility of applying a mask to filter
# If a mask is passed on, only the pixels inside the mask are used to compute the mean colors
def get_mean_colors(collection, mask=None):
    # Determine if the collecttion contains grayscale images or rgb. All must be the same
    is_grayscale = len(collection[0].shape) == 2
    mean_colors = []

    if mask is None:
        # Build a mask full of True values for compatibility
        mask = np.full(collection[0].shape, True)

    for image in collection:
        if is_grayscale:
            mean_color = image[mask].mean()
        else:
            pixels = image[mask].reshape(image[mask].shape[0] // 3, 3)
            mean_color = [pixels[:, i].mean() for i in range(3)]
        mean_colors.append(mean_color)

    return np.asarray(mean_colors)


# Get the image with the closest color to a given patch
def get_closest_image(collection, colors, patch, exclude):
    # Determine if the collection is composed of grayscale images
    is_grayscale = len(collection[0].shape) == 2
    # Create a mask to exclude neighbour pics
    mask = np.full(len(collection), True)
    if exclude:
        mask[exclude] = False

    # Flatten the patch
    patch = patch.reshape(-1, patch.shape[-1])
    # Determine the mean color of the patch
    if is_grayscale:
        color = patch.mean()
    else:
        color = [patch[:, i].mean() for i in range(3)]

    if is_grayscale:
        distances = np.abs(colors[mask] - color)
    else:
        # Compute the euclidian distances between the colors and the given color
        distances = np.linalg.norm(colors[mask] - color, axis=-1)

    ind = distances.argmin()
    # Find the corresponding index in the initial collection based on the one obtained after removing
    # elements
    for idx in sorted(set(exclude)):
        if idx <= ind:
            ind += 1

    # Return the picture with the minimum distance and the corresponding index
    return collection[ind], ind


def get_random_image(collection, exclude):
    # Create a mask to exclude neighbour pics
    mask = np.full(len(collection), True)
    if exclude:
        mask[exclude] = False

    ind = randint(0, collection[mask].shape[0] - 1)
    # Find the corresponding index in the initial collection based on the one obtained after removing
    # elements
    for idx in sorted(set(exclude)):
        if idx <= ind:
            ind += 1

    return collection[ind], ind


# noinspection PyUnboundLocalVariable
def build_grid_mosaic(image, collection):
    # Get the mean color of each image in collection
    mean_colors = get_mean_colors(collection)
    # Get the pictures reference size
    pic_size = collection[0].shape
    # Create a 2D array to retain the neighbour patches used
    if different_neighbours:
        neighbours = np.empty((image.shape[0] // pic_size[0], image.shape[1] // pic_size[1]), dtype=np.uint)
    # Create the mosaic
    mosaic = np.empty_like(image)

    for i in range(0, image.shape[0], pic_size[0]):
        for j in range(0, image.shape[1], pic_size[1]):
            # Determine which pics should be excluded
            exclude = []
            if different_neighbours:
                grid_line = i // pic_size[0]
                grid_col = j // pic_size[1]
                if grid_line > 0:
                    exclude.append(neighbours[grid_line - 1, grid_col])
                if grid_col > 0:
                    exclude.append(neighbours[grid_line, grid_col - 1])

            if criteria == 'mean_color_distance':
                patch = image[i:i + pic_size[0], j:j + pic_size[1]]
                # Get the image with the closest mean color
                pic, ind = get_closest_image(collection, mean_colors, patch, exclude)
            else:
                pic, ind = get_random_image(collection, exclude)

            # Attach the patch on the mosaic
            mosaic[i:i + pic_size[0], j:j + pic_size[1]] = pic

            if different_neighbours:
                # Remember what patch has been used
                neighbours[grid_line, grid_col] = ind

    return mosaic


def build_random_mosaic(image, collection):
    # Determine if the image is grayscale
    is_grayscale = len(image.shape) == 2
    # Get the mean color of each image in collection
    mean_colors = get_mean_colors(collection)
    # Get the pictures reference size
    patch_size = collection[0].shape
    # Compute how many pixels need to be filled
    to_be_filled = np.prod(image.shape)
    # Create an empty grid for the mosaic
    mosaic = np.full(image.shape, -1, dtype=np.int16)
    start_time = time.time()
    pixels_number = image.shape[0] * image.shape[1]

    # Random fill the mosaic until X(treshold) percent of the image remained unfilled
    while to_be_filled / pixels_number * 100 > treshold:
        # Pic a random position on the image
        i = randint(0, image.shape[0] - 1)
        j = randint(0, image.shape[1] - 1)
        # Check to which extend we can go when selecting the patch
        to_i = min(image.shape[0], i + patch_size[0])
        to_j = min(image.shape[1], j + patch_size[1])

        # Replace only if there are unfilled pixels
        if -1 in mosaic[i:to_i, j:to_j]:
            if criteria == 'mean_color_distance':
                patch = image[i:to_i, j:to_j]
                # Get the image with the closest mean
                pic, _ = get_closest_image(collection, mean_colors, patch, [])
            else:
                pic, _ = get_random_image(collection, [])

            # Subtract the number of pixels filled
            to_be_filled -= np.sum(mosaic[i:to_i, j:to_j] == -1)
            # Determine how much of the pic to be used when filling
            height = to_i - i
            width = to_j - j
            # Attach the pic on the mosaic
            mosaic[i:to_i, j:to_j] = pic[:height, :width]
    print(str(100 - treshold) + "% of image completed in: " + str(time.time() - start_time) + " seconds.")

    # Find the positions of unfilled pixels
    if is_grayscale:
        (rows, columns) = np.where(mosaic == -1)
    else:
        (rows, columns) = np.where((mosaic[:, :, 0] == -1))

    for i in range(len(rows)):
        line, col = rows[i], columns[i]
        if (is_grayscale and mosaic[line, col] == -1) or (not is_grayscale and mosaic[line, col, 0] == -1):
            # Determine how much of the patch is inside the image
            to_line = min(image.shape[0], line + patch_size[0])
            to_col = min(image.shape[1], col + patch_size[1])

            if criteria == 'mean_color_distance':
                patch = image[line:to_line, col:to_col]
                # Get the image with the closest mean
                pic, _ = get_closest_image(collection, mean_colors, patch, [])
            else:
                pic, _ = get_random_image(collection, [])

            # Determine how much of the pic to be used when filling
            height = to_line - line
            width = to_col - col
            # Attach the pic on the mosaic
            mosaic[line:to_line, col:to_col] = pic[:height, :width]

    return mosaic


# Get hex keypoints
def get_hex_points(size):
    mid_line = (size[0] + 1) // 2  # exclusive upper bound
    top_left_corner = (size[0] - 1) // 2
    top_right_corner = size[1] - mid_line

    return mid_line, top_left_corner, top_right_corner


# Get a hex mask for a given size
def get_hex_mask(size):
    # Get the keypoints of the hex
    mid_line, start_column, end_column = get_hex_points(size)
    hex_mask = np.full(size, False)

    # Create the top half of the hex
    for i in range(mid_line):
        hex_mask[i, start_column - i:end_column + 1 + i] = True
    # Flip the top half and add it to the lower half
    hex_mask[mid_line:] = np.flipud(hex_mask[:mid_line])

    return hex_mask


# Remove the padding of an image, based on it's initial size
def remove_padding(image, old_size):
    left_padding = (image.shape[1] - old_size[1]) // 2
    top_padding = (image.shape[0] - old_size[0]) // 2
    from_line, to_line = top_padding, image.shape[0] - top_padding
    from_column, to_column = left_padding, image.shape[1] - left_padding

    return image[from_line:to_line, from_column:to_column]


# Create the mosaic using hex pics
# noinspection PyUnboundLocalVariable
def build_hex_mosaic(image, collection):
    pic_shape = collection[0].shape
    is_grayscale = len(image.shape) == 2
    # Get a 3d hex mask for pictures, used when placing patches on the grid
    mask = get_hex_mask(pic_shape)

    # Crop the collection's pics into hex shapes
    collection = collection * mask
    # Get the mean colors of the collection
    mean_colors = get_mean_colors(collection, mask)

    _, top_left_corner, top_right_corner = get_hex_points(pic_shape)
    top_edge_size = top_right_corner - top_left_corner + 1
    old_shape = image.shape

    # Add a padding to the image on each side to ease the work. The padded pixels will be symmetric along the edge.
    left_padding = top_right_corner
    top_padding = pic_shape[0] // 2
    if is_grayscale:
        image = np.pad(image, ((top_padding, top_padding), (left_padding, left_padding)), 'symmetric')
    else:
        image = np.pad(image, ((top_padding, top_padding), (left_padding, left_padding), (0, 0)), 'symmetric')

    mosaic = np.empty_like(image, dtype=np.uint8)
    # A dictionary to retain the image used for each patch
    neighbours = {}
    row_pace = pic_shape[0] // 2

    # The anchor point of each hex image is the top-left corner of the square image
    for i in range(0, mosaic.shape[0], row_pace):
        # We want to offset the starting column based on what row we are
        row_number = i // row_pace
        start_column = ((row_number + 1) % 2) * (top_right_corner + 1)

        # The length of a pic plus the top edge before the next patch on the same line
        column_pace = pic_shape[1] + top_edge_size
        # Taking care not to exit the mosaic when slicing
        to_i = min(mosaic.shape[0], i + pic_shape[0])
        # Height of the hex pic that fits the mosaic
        height = to_i - i

        for j in range(start_column, mosaic.shape[1], column_pace):
            # Taking care not to exit the mosaic when slicing
            to_j = min(mosaic.shape[1], j + pic_shape[1])
            # Width of the hex pic that fits the mosaic
            width = to_j - j

            # See what neighbours the patch has
            exclude = []
            if different_neighbours:
                # Get the patch right above
                if i >= pic_shape[0]:
                    exclude.append(neighbours[(i - pic_shape[0], j)])
                # Get the up-right and up-left patches
                if i >= row_pace:
                    if j >= column_pace:
                        # Get the left patch
                        exclude.append(neighbours[i, j - column_pace])
                        exclude.append(neighbours[i - row_pace, j - (top_right_corner + 1)])
                    if j < image.shape[1] - column_pace:
                        exclude.append(neighbours[i - row_pace, j + top_right_corner + 1])
                # Also get the left patch if we are on the first line
                elif j >= column_pace:
                    exclude.append(neighbours[i, j - column_pace])

            # Select a patch in the mosaic
            selected_patch = image[i:to_i, j:to_j]
            # Crop the mask to fit the cropped patch
            fitted_mask = mask[:height, :width]

            if criteria == 'mean_color_distance':
                patch = selected_patch[fitted_mask]
                # Get the hex pic from collection that has the closest mean color
                pic, ind = get_closest_image(collection, mean_colors, patch, exclude)
            else:
                pic, ind = get_random_image(collection, exclude)

            if different_neighbours:
                # Remember what pic has been used
                neighbours[(i, j)] = ind

            # Replace the hex patch in the mosaic with the hex pic from collection
            mosaic[i:to_i, j:to_j][fitted_mask] = pic[:height, :width][fitted_mask]

    return remove_padding(mosaic, old_shape)


# Read and resize the image
print("Reading image...")
image = np.asarray(Image.open(image_path))

is_grayscale = len(image.shape) == 2

# Read the image collection
print("Reading pictures collection...")
collection = read_collection(collection_dir, mime_type, is_grayscale)

# # Read CIFAR-10 data batch
# images, labels = unpickle('./data/data_batch_1')
# # Get only one class of pictures
# images = images[labels == 8]
# # Compute the number of lines and columns of the image
# image_size = int((images.shape[1] / 3) ** (1 / 2))
# # Reshape each image that it contains three 2D arrays, each one representing one RGB channel
# images = images.reshape((images.shape[0], 3, image_size, image_size))
# # Overlay the three channels to create the RGB image
# collection = np.stack((images[:, 0], images[:, 1], images[:, 2]), 3)

print("Computing new image sizing for " + str(horizontal_pieces) + " pieces horizontally...")
new_size = get_new_size(image.shape, horizontal_pieces, collection[0].shape)
print("New size: " + str(new_size[1]) + "x" + str(new_size[0]) + "px. Resizing image...")
image = transform.resize(image, new_size)
# Convert image from float64 to uint8, truncating the results
image = (image * 255).astype(np.uint8)

# Display a preview of the collection
if show_collection:
    print("Showing a preview of the collection...")
    preview = get_preview(collection)
    plt.figure()
    plt.imshow(preview, cmap='gray' if is_grayscale else None)
    plt.show()

    # Save image to file
    # misc.imsave('preview.jpeg', preview)

print("Building the mosaic...")
start = time.time()
if arrange_mode is 'grid':
    mosaic = build_grid_mosaic(image, collection)
else:
    mosaic = build_random_mosaic(image, collection).astype(np.uint8)
print("Mosaic created in: " + str(time.time() - start) + " seconds.")
plt.figure()
plt.imshow(mosaic, cmap='gray' if is_grayscale else None)
plt.show()

# Save image to file
# misc.imsave('imagine.jpeg', mosaic)

print("Building the mosaic using hexagonal pics...")
start = time.time()
mosaic = build_hex_mosaic(image, collection)
plt.imshow(mosaic, cmap='gray' if is_grayscale else None)
plt.show()
print("Mosaic created in: " + str(time.time() - start) + " seconds.")

# Save image to file
# misc.imsave('imagine-hex.jpeg', mosaic)
