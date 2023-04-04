import cv2
import numpy as np
import matplotlib.pyplot as plt

img_file = "./lion.jpg"
key = 94398

def shuffle_pixels(img, seed):
    flat_size = img.size
    np.random.seed(seed)
    R1 = np.random.permutation(flat_size)
    R2 = np.random.permutation(flat_size)

    flattened = img.flatten()
    result = flattened.copy()

    for i in range(flat_size):
        temp = result[R1[i]]
        result[R1[i]] = result[R2[i]]
        result[R2[i]] = temp

    return result.reshape(img.shape)

def unshuffle_pixels(img, seed):
    flat_size = img.size
    np.random.seed(seed)
    R1 = np.random.permutation(flat_size)
    R2 = np.random.permutation(flat_size)

    flattened = img.flatten()
    result = flattened.copy()

    for i in range(flat_size - 1, -1, -1):
        temp = result[R1[i]]
        result[R1[i]] = result[R2[i]]
        result[R2[i]] = temp

    return result.reshape(img.shape)

# Read in grayscale image
img = cv2.imread(f"{img_file}", cv2.IMREAD_GRAYSCALE)

# Set order of bit planes using a randomly generated ordering set (in this case, seed is constant)
np.random.seed(key)  # 938616836
set = np.random.permutation(8)
print("Permutations: ", set)

# Decompose image, shuffle pixel locations, and shuffle bit planes
bit_planes = []
for i in range(0, 8):
    bit_plane = (img >> set[i]) & 1
    shuffled_bit_plane = shuffle_pixels(bit_plane, seed=i)
    bit_planes.append(shuffled_bit_plane)

# for i in range(0, 8): 
#     bit_plane = ((img >> i) & 1) * 255
#     bit_planes.append(bit_plane)


# for i, bit_plane in enumerate(bit_planes):
#     cv2.imshow(f"Bit-Plane #{i + 1}", bit_plane*255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Construct scrambled image
scrambled_img = np.zeros_like(img)
for i in range(len(bit_planes)):
    scrambled_img += bit_planes[i] << i

cv2.imwrite("ENCRYPTED_IMAGE.png", scrambled_img)

# Unscramble/unshuffle bitplanes
bit_planes.clear()
bit_planes = [None] * 8

for i in range(0, 8):
    bit_plane = (scrambled_img >> i) & 1
    unshuffled_bit_plane = unshuffle_pixels(bit_plane, seed=i)
    bit_planes[set[i]] = unshuffled_bit_plane

# Reconstruct image from properly ordered bit planes
reconstructed_img = np.zeros_like(img)
for i in range(len(bit_planes)):
    reconstructed_img += bit_planes[i] << i

def display_image_and_histogram(img, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(img, cmap='gray')
    ax1.set_title(title)
    ax1.axis('off')

    ax2.hist(img.ravel(), bins=256, range=(0, 256))
    ax2.set_title(f"{title} Histogram")
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Count')
    plt.tight_layout()

def display_difference_histogram(img1, img2, title):
    diff_img = np.abs(img1 - img2)
    plt.figure(figsize=(6, 4))
    plt.hist(diff_img.ravel(), bins=256, range=(0, 256))
    plt.title(title)
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.tight_layout()

# Display the difference histogram
# display_difference_histogram(img, reconstructed_img, "Difference Histogram")
# plt.show()


# Display images and histograms
display_image_and_histogram(img, "Original Image")
display_image_and_histogram(scrambled_img, "Encrypted Image")
display_image_and_histogram(reconstructed_img, "Reconstructed Image")

# Show all figures simultaneously
plt.show()
