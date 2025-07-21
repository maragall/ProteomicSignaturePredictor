import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.stats import zscore
from skimage.exposure import adjust_gamma
from skimage.filters import threshold_local
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import tiffslide as openslide
def coordinate_pairs(v1,v2):
    for i in v1:
        for j in v2:
            yield i,j
def deconvolution(img,MOD):
    MODx=MOD[0]
    MODy=MOD[1]
    MODz=MOD[2]
    cosx=np.zeros((3,))
    cosy=np.zeros((3,))
    cosz=np.zeros((3,))
    len=np.zeros((3,))
    for i in range(0,3):
        cosx[i]=cosy[i]=cosz[i]=0.0
        len[i]=np.sqrt(MODx[i]*MODx[i] + MODy[i]*MODy[i] + MODz[i]*MODz[i])
        if len[i]!=0.0:
            cosx[i]= MODx[i]/len[i]
            cosy[i]= MODy[i]/len[i]
            cosz[i]= MODz[i]/len[i]

    if cosx[1]==0.0:
        if cosy[1]==0.0:
            if cosz[1]==0.0:
                cosx[1]=cosz[0]
                cosy[1]=cosx[0]
                cosz[1]=cosy[0]

    if cosx[2]==0.0:
    	if cosy[2]==0.0:
    		if cosz[2]==0.0:
    			if ((cosx[0]*cosx[0] + cosx[1]*cosx[1])> 1):
    				cosx[2]=0.0
    			else:
    				cosx[2]=np.sqrt(1.0-(cosx[0]*cosx[0])-(cosx[1]*cosx[1]))

    			if ((cosy[0]*cosy[0] + cosy[1]*cosy[1])> 1):
    				cosy[2]=0.0
    			else:
    				cosy[2]=np.sqrt(1.0-(cosy[0]*cosy[0])-(cosy[1]*cosy[1]))

    			if ((cosz[0]*cosz[0] + cosz[1]*cosz[1])> 1):
    				cosz[2]=0.0
    			else:
    				cosz[2]=np.sqrt(1.0-(cosz[0]*cosz[0])-(cosz[1]*cosz[1]))
    leng= np.sqrt(cosx[2]*cosx[2] + cosy[2]*cosy[2] + cosz[2]*cosz[2])

    cosx[2]= cosx[2]/leng
    cosy[2]= cosy[2]/leng
    cosz[2]= cosz[2]/leng


    A = cosy[1] - cosx[1] * cosy[0] / cosx[0]
    V = cosz[1] - cosx[1] * cosz[0] / cosx[0]
    C = cosz[2] - cosy[2] * V/A + cosx[2] * (V/A * cosy[0] / cosx[0] - cosz[0] / cosx[0])
    q=np.zeros((9,))
    q[2] = (-cosx[2] / cosx[0] - cosx[2] / A * cosx[1] / cosx[0] * cosy[0] / cosx[0] + cosy[2] / A * cosx[1] / cosx[0]) / C;
    q[1] = -q[2] * V / A - cosx[1] / (cosx[0] * A);
    q[0] = 1.0 / cosx[0] - q[1] * cosy[0] / cosx[0] - q[2] * cosz[0] / cosx[0];
    q[5] = (-cosy[2] / A + cosx[2] / A * cosy[0] / cosx[0]) / C;
    q[4] = -q[5] * V / A + 1.0 / A;
    q[3] = -q[4] * cosy[0] / cosx[0] - q[5] * cosz[0] / cosx[0];
    q[8] = 1.0 / C;
    q[7] = -q[8] * V / A;
    q[6] = -q[7] * cosy[0] / cosx[0] - q[8] * cosz[0] / cosx[0];

    img_stain1 = np.ravel(np.copy(img[:,:,0]))
    img_stain2 = np.ravel(np.copy(img[:,:,1]))
    img_stain3 = np.ravel(np.copy(img[:,:,2]))
    dims=img.shape
    imagesize = dims[0] * dims[1]
    rvec=np.ravel(np.copy(img[:,:,0])).astype('float')
    gvec=np.ravel(np.copy(img[:,:,1])).astype('float')
    bvec=np.ravel(np.copy(img[:,:,2])).astype('float')
    log255=np.log(255.0)
    for i in range(0,imagesize):
        R = rvec[i]
        G = gvec[i]
        B = bvec[i]

        Rlog = -((255.0*np.log((R+1)/255.0))/log255)
        Glog = -((255.0*np.log((G+1)/255.0))/log255)
        Blog = -((255.0*np.log((B+1)/255.0))/log255)
        for j in range(0,3):
            Rscaled = Rlog * q[j*3];
            Gscaled = Glog * q[j*3+1];
            Bscaled = Blog * q[j*3+2];

            output = np.exp(-((Rscaled + Gscaled + Bscaled) - 255.0) * log255 / 255.0)
            if(output>255):
                output=255

            if j==0:
                img_stain1[i] = np.floor(output+.5)
            elif j==1:
                img_stain2[i] = np.floor(output+.5)
            else:
            	img_stain3[i] = np.floor(output+.5)
    img_stain1=np.reshape(img_stain1,(dims[0],dims[1]))
    img_stain2=np.reshape(img_stain2,(dims[0],dims[1]))
    img_stain3=np.reshape(img_stain3,(dims[0],dims[1]))
    return img_stain1,img_stain2,img_stain3



def deconvolution_WSI(img_path,MOD,block_size):
    slide=openslide.OpenSlide(img_path)
    dim_x,dim_y=slide.dimensions
    # slide.close()
    index_x=np.array(range(0,dim_x,block_size))
    index_y=np.array(range(0,dim_y,block_size))
    img_stain1=np.zeros((dim_y,dim_x))
    img_stain2=np.zeros((dim_y,dim_x))
    img_stain3=np.zeros((dim_y,dim_x))
    totalpatches=len(index_x)*len(index_y)

    # deconv_images=Parallel(n_jobs=multiprocessing.cpu_count())(delayed(deconvolution)(np.array(slide.read_region((i,j),
    #     0,(min(dim_y,i+block_size)-i,min(dim_x,j+block_size)-j)))[:,:,:3],MOD) for i in tqdm(index_y) for j in index_x)
    deconv_images=Parallel(n_jobs=multiprocessing.cpu_count())(delayed(deconvolution)(np.array(slide.read_region((j,i),
        0,(min(dim_x,j+block_size)-j,min(dim_y,i+block_size)-i)))[:,:,:3],MOD) for i in tqdm(index_y) for j in index_x)

    # with tqdm(total=totalpatches,unit='image',colour='green',desc='Total WSI progress') as pbar:
    #     for i,j in coordinate_pairs(index_y,index_x):
    counter=0
    for i in tqdm(index_y,desc='Deconvolving hematoxylin...'):
        for j in index_x:
            yEnd = min(dim_y,i+block_size)
            xEnd = min(dim_x,j+block_size)

            xLen=xEnd-j
            yLen=yEnd-i

            dxS=j
            dyS=i
            dxE=j+xLen
            dyE=i+yLen

            img_stain1[dyS:dyE,dxS:dxE]=deconv_images[counter][0]
            img_stain2[dyS:dyE,dxS:dxE]=deconv_images[counter][1]
            img_stain3[dyS:dyE,dxS:dxE]=deconv_images[counter][2]

            # pbar.update(1)
            counter+=1
    return img_stain1,img_stain2,img_stain3









    import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.stats import zscore
from skimage.exposure import adjust_gamma
from skimage.filters import threshold_local
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# The functions (coordinate_pairs, deconvolution, deconvolution_WSI) should be defined here as in your original script

# Set up MOD matrix (example values, you should use the appropriate values for your stains)
MODx = np.zeros((3,))
MODy = np.zeros((3,))
MODz = np.zeros((3,))

MODx[0] = 0.651 #Hematoxilyn
MODy[0] = 0.701
MODz[0] = 0.210


MODx[1] = 0.216 #Eosin
MODy[1] = 0.801
MODz[1] = 0.568

MODx[2] = 0.316 #Residual
MODy[2] = -0.598
MODz[2] = 0.737

MOD = [MODx, MODy, MODz]

# Path to the WSI
image_path ="/blue/pinaki.sarder/j.maragall/CellMapping/15-1 Stitch.tif"

# Block size (e.g. 512)
block_size = 512

# Perform deconvolution
img_stain1, img_stain2, img_stain3 = deconvolution_WSI(image_path, MOD, block_size)









import numpy as np

# Assuming img_stain1 is a numpy array containing the image data
# And assuming you want to set an arbitrary threshold at the mean value you provided

threshold_value = 100

binary_mask = np.where(img_stain1 < threshold_value, 1, 0)

# Now, binary_mask contains 1 where the intensity in img_stain1 was above the threshold, and 0 elsewhere.









import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread

# Assuming binary_mask is your binary mask array, as created previously

# Save the binary mask to a .tiff file
output_file_path = 'R2_binary_mask.tiff'
imsave(output_file_path, (binary_mask.astype(np.uint8) * 255))

# Read the saved .tiff file
loaded_binary_mask = imread(output_file_path) / 255









import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from skimage.measure import label, regionprops

# Assuming binary_mask is your binary mask array, as created previously

# Label connected regions
label_image = label(binary_mask)

# Iterate through the regions and set those with area > 3000 to black. this takes out blobs (of dirt)
for region in regionprops(label_image):
    if region.area > 3000:
        # Set the region to black
        coords = region.coords
        binary_mask[coords[:, 0], coords[:, 1]] = 0
        
# Iterate through the regions and set those with area < 10 to black. This takes out specs (of dirt)
for region in regionprops(label_image):
    if region.area < 10:
        # Set the region to black
        coords = region.coords
        binary_mask[coords[:, 0], coords[:, 1]] = 0


# Save the modified binary mask to a .tiff file
output_file_path = 'R2_HE_modified_binary_mask.tiff'
imsave(output_file_path, (binary_mask.astype(np.uint8) * 255))

# Read the saved .tiff file
loaded_binary_mask = imread(output_file_path) / 255

# Visualize the loaded binary mask
plt.imshow(loaded_binary_mask, cmap='gray')
plt.title("Modified Binary Mask")
plt.axis('off')
plt.show()









import cv2
import numpy as np
import matplotlib.pyplot as plt
import tiffslide as openslide
from tifffile import imread
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy
from joblib import Parallel, delayed
import pandas as pd
from umap import UMAP









import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Assuming 'DAPI_mask' is your DAPI mask image
# Assuming 'object_cluster_labels' is the dictionary mapping cluster labels to objects

# Create a copy of the DAPI mask to store the colored objects
colored_mask = np.zeros_like(DAPI_mask, dtype=np.uint8)

# Iterate over each object in the DAPI mask
for region_label in range(1, number_of_objects + 1):
    # Get the cluster label for the object
    cluster_label = object_cluster_labels[region_label]

    # Assign a color to the object based on the cluster label
    color = cm.viridis(cluster_label / 10)[:3] * 255  # Adjust the colormap as per your needs

    # Create a binary mask for the current object
    object_mask = DAPI_mask == region_label

    # Reshape the color array to match the shape of the object mask
    color = np.reshape(color, (1, 1, 3))

    # Color the object in the colored mask using the binary mask
    colored_mask = np.where(object_mask[..., None], color, colored_mask)

# Display the colored mask
plt.imshow(colored_mask)
plt.axis('off')
plt.show()









def normalize_image(image, target_dtype=np.uint8):
    normalized_image = cv2.normalize(image, None, 0, np.iinfo(target_dtype).max, cv2.NORM_MINMAX)
    return normalized_image.astype(target_dtype)



# Read the images using OpenSlide
image1_path_transformee = "/home/j.maragall/Pipeline_CellMapping/R2_HE_modified_binary_mask.tiff" # "/blue/pinaki.sarder/j.maragall/CellMapping/15-1 Stitch.tif"
image2_path_anchor = "/home/j.maragall/Pipelne_CellMapping/R2_DAPI_cleaned_mask.tif"

slide1 = openslide.OpenSlide(image1_path_transformee)
slide2 = openslide.OpenSlide(image2_path_anchor)

# Convert the images to NumPy arrays and normalize them to uint8
img1 = normalize_image(np.array(slide1.get_thumbnail(slide1.dimensions)), np.uint8)
img2 = normalize_image(np.array(slide2.get_thumbnail(slide2.dimensions)), np.uint8)

# Visualize the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title("Image 1 (DAPI Binary Mask)")

plt.subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title("Image 2 (HE Modified Binary Mask)")

plt.show()



# Create SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors using SIFT
kp1, d1 = sift.detectAndCompute(img1, None)
kp2, d2 = sift.detectAndCompute(img2, None)

# Flann-based matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Find matches using KNN
matches = matcher.knnMatch(d1, d2, k=2)

# Filter matches using the ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# ... rest of the code for finding homography, warping and computing IoU


# Define empty matrices
p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find the homography matrix
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

# Use this matrix to transform the binary mask wrt the reference binary mask
height, width = img2.shape[:2]
transformed_mask = cv2.warpPerspective(img1, homography, (width, height))









# Visualize the transformed mask
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title("Image 1 (DAPI Binary Mask)")

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.title("Image 2 (HE Modified Binary Mask)")

plt.subplot(1, 3, 3)
plt.imshow(transformed_mask, cmap='gray')
plt.title("Transformed Mask")

plt.show()









image3_path = "/blue/pinaki.sarder/j.maragall/CellMapping/15-1 Stitch.tif"
slide3 = openslide.OpenSlide(image3_path)

img3 = normalize_image(np.array(slide3.get_thumbnail(slide3.dimensions)), np.uint8)

transformed_img3 = cv2.warpPerspective(img3, homography, (width, height))

plt.imshow(transformed_img3, cmap='gray')
plt.title("Transformed Image 3")
plt.show()



# Specify the path and filename for saving the transformed image
output_path = "./transformed_image.tif"

# Save the transformed image as a .tif file
cv2.imwrite(output_path, transformed_img3)







import tiffslide as openslide
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imsave
import cv2
import imageio
from tifffile import imread









def normalize_CODEX(image):
  """Normalizes an image to uint8.

  Args:
    image: The image to be normalized.

  Returns:
    A normalized image in uint8 format.
  """

  max_value = np.iinfo(image.dtype).max
  image = image / max_value
  image = image * 255
  image = image.astype(np.uint8)

  return image

read_CODEX = imread("/blue/pinaki.sarder/j.maragall/CellMapping/15-1.tif")
CODEX = normalize_CODEX(read_CODEX)









def normalize_mask(image, target_dtype=np.uint8):
    normalized_image = cv2.normalize(image, None, 0, np.iinfo(target_dtype).max, cv2.NORM_MINMAX)
    return normalized_image.astype(target_dtype)



read_DAPI = openslide.open_slide("/home/j.maragall/Pipeline_CellMapping/R2_DAPI_cleaned_mask.tif")
DAPI_mask = normalize_mask(np.array(read_DAPI.get_thumbnail(read_DAPI.dimensions)), np.uint8)









def visualize_channels(image, channel):
    """
    Get the channel index based on the lowest value in the tuple and visualize the image based on where the channels actually are.

    Args:
        image: The image to visualize.
    """

    min_value = min(image.shape)

    if min_value == image.shape[0]:
        # The channels are in the first dimension
        slide = image[channel]
        plt.imshow(slide, cmap="gray")
        plt.title(f"Channel {channel}")
        plt.show()
    elif min_value == image.shape[1]:
        # The channels are in the second dimension
        slide = image[:, channel]
        plt.imshow(slide, cmap="gray")
        plt.title(f"Channel {channel}")
        plt.show()
    else:
        # The channels are in the third dimension
        slide = image[:, :, channel]
        plt.imshow(slide, cmap="gray")
        plt.title(f"Channel {channel}")
        plt.show()

#visualize
visualize_channels(CODEX,6)





def position_channels(image): # Get the channel index based on the lowest value in the tuple and visualize the image based on where the channels actually are.

    min_value = min(image.shape)

    if min_value == image.shape[0]:
        position = 0
    elif min_value == image.shape[1]:
        position = 1
    else:
        position = 2
    return position

def single_channel(slide, channel_index): #stores the single channel of an image, independently from where the channel value is in the imag.shape tuple
    location = position_channels(slide)
    
    if location == 0:
        image = slide[channel_index]
    elif location == 1:
        image = slide[:, channel_index]
    else:
        image = slide[:, :, channel_index]
    return image









CODEX.shape[position_channels(CODEX)]









import cv2
import numpy as np
import matplotlib.pyplot as plt
import tiffslide as openslide
from tifffile import imread
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy
from joblib import Parallel, delayed
import pandas as pd
from umap import UMAP


# Now let's extract features from each object in the binary mask
# First, label each object
labeled_image = label(DAPI_mask)
number_of_objects = labeled_image.max()
print(number_of_objects) #check if quantity is coherent











# Optimized function to extract features from a single object
def extract_features(region_label):
    object_mask = (labeled_image == region_label)

    # Get the properties of the object
    properties = regionprops(object_mask.astype(int))[0]
    area = properties.area
    perimeter = properties.perimeter
    eccentricity = properties.eccentricity

    channel_features = np.zeros(4 * CODEX.shape[position_channels(CODEX)] + 3, dtype=np.float32)

    # Extract the channel and calculate the features
    for channel_index in range(CODEX.shape[position_channels(CODEX)]):
        single_channel_img = single_channel(CODEX, channel_index) 
        masked_single_channel = single_channel_img * object_mask

        channel_data = masked_single_channel[object_mask]

        mean_intensity = np.mean(channel_data)
        std_intensity = np.std(channel_data)
        variance_intensity = np.var(channel_data)
        hist, _ = np.histogram(channel_data, bins=256, range=(0, 256), density=True)
        entropy_intensity = entropy(hist)

        index = channel_index * 4
        channel_features[index] = mean_intensity
        channel_features[index + 1] = std_intensity
        channel_features[index + 2] = variance_intensity
        channel_features[index + 3] = entropy_intensity

    # Storing object features at the end
    channel_features[-3] = area
    channel_features[-2] = perimeter
    channel_features[-1] = eccentricity

    # Return the object features
    return channel_features


# Parallel feature extraction
num_channels = CODEX.shape[position_channels(CODEX)]
num_features_per_channel = 4  # mean, std, variance, entropy
num_object_features = 3  # area, perimeter, eccentricity

# Pre-allocate memory
features_array = np.zeros((number_of_objects, num_features_per_channel * num_channels + num_object_features), dtype=np.float32)

# Parallel extraction
features_list = Parallel(n_jobs=18)(delayed(extract_features)(region_label) for region_label in range(1, number_of_objects + 1))

# Copy the features from the list to the pre-allocated NumPy array
for i, features in enumerate(features_list):
    features_array[i, :] = features

# Convert to dataframe
columns = [f'channel_{i}_mean' for i in range(num_channels)] + [f'channel_{i}_std' for i in range(num_channels)] + \
          [f'channel_{i}_variance' for i in range(num_channels)] + [f'channel_{i}_entropy' for i in range(num_channels)] + \
          ['area', 'perimeter', 'eccentricity']

features_df = pd.DataFrame(features_array, columns=columns)

import pandas as pd

features_df.to_csv('features.csv', index=False)









from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from scipy.stats import skew, kurtosis
from skimage.util import img_as_ubyte
from skimage import img_as_float

# Optimized function to extract new features from a single object
def extract_new_features(region_label):
    object_mask = (labeled_image == region_label)

    # Extract the channel and calculate the features
    new_channel_features = np.zeros(7 * CODEX.shape[position_channels(CODEX)], dtype=np.float32)
    for channel_index in range(CODEX.shape[position_channels(CODEX)]):
        single_channel_img = single_channel(CODEX, channel_index) 
        masked_single_channel = single_channel_img * object_mask

        channel_data = masked_single_channel[object_mask]

        try:
            # New features
            contrast = np.ptp(channel_data)
            energy = np.sum(channel_data**2)
            rms = np.sqrt(np.mean(channel_data**2))
            smoothness = 1 - 1 / (1 + np.var(channel_data))
            uniformity = np.sum(channel_data**2) / (np.sqrt(np.mean(channel_data**2)) + np.finfo(float).eps)
            skewness = skew(channel_data)
            kurt = kurtosis(channel_data)

            index = channel_index * 7
            new_channel_features[index:index+7] = [contrast, energy, rms, smoothness, uniformity, skewness, kurt]

        except Exception as e:
            print(f"Failed to extract features for region {region_label}, channel {channel_index}. Error: {e}")

    # Return the new object features
    return new_channel_features

# Parallel extraction
num_channels = CODEX.shape[position_channels(CODEX)]
num_new_features_per_channel = 7  # contrast, energy, rms, smoothness, uniformity, skewness, kurtosis

# Pre-allocate memory
new_features_array = np.zeros((number_of_objects, num_new_features_per_channel * num_channels), dtype=np.float32)

# Parallel extraction
new_features_list = Parallel(n_jobs=11)(delayed(extract_new_features)(region_label) for region_label in range(1, number_of_objects + 1))

# Copy the features from the list to the pre-allocated NumPy array
for i, features in enumerate(new_features_list):
    new_features_array[i, :] = features

# Convert to dataframe
new_columns = [f'channel_{i}_contrast' for i in range(num_channels)] + [f'channel_{i}_energy' for i in range(num_channels)] + \
          [f'channel_{i}_rms' for i in range(num_channels)] + [f'channel_{i}_smoothness' for i in range(num_channels)] + \
          [f'channel_{i}_uniformity' for i in range(num_channels)] + [f'channel_{i}_skewness' for i in range(num_channels)] + \
          [f'channel_{i}_kurtosis' for i in range(num_channels)] 

new_features_df = pd.DataFrame(new_features_array, columns=new_columns)

# Join the new features to the existing ones
final_df = pd.concat([features_df, new_features_df], axis=1)

# Save the final DataFrame to a new .csv file
final_df.to_csv('final_features.csv', index=False)









import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Specify the path to the CSV file
csv_file_path = "/home/j.maragall/final_features.csv"

# Load the data from the CSV file
features_df = pd.read_csv(csv_file_path)

# Impute all NaN values with 0
features_df.fillna(0, inplace=True)

# Create a scaler object
scaler = MinMaxScaler()

# Fit and transform the data
features_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)



















import pandas as pd
from sklearn.preprocessing import MinMaxScaler



# Specify the percentage threshold for dropping
zero_threshold = 0.97

# Create a mask of boolean values representing whether each value in the DataFrame is zero
is_zero = features_df == 0

# Calculate the percentage of zeros in each column
zero_percentage = is_zero.mean()

# Identify the columns which have more than 97% zeros
columns_to_drop = zero_percentage[zero_percentage > zero_threshold].index

# Drop those columns from the DataFrame
features_df = features_df.drop(columns=columns_to_drop)

# Print the updated DataFrame
features_df.head()












import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Specify the path to the image
image1_path_transformee = "/home/j.maragall/Pipeline_CellMapping/R2_DAPI_cleaned_mask.tif"

# Read and display the image
image1_transformee = mpimg.imread(image1_path_transformee)
plt.imshow(image1_transformee, cmap='gray')
plt.axis('off')
plt.title('Image 1')
plt.show()









import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Specify the path to the image
image2_path_anchor = "/blue/pinaki.sarder/j.maragall/CellMapping/15-1 Stitch.tif"

# Read and display the image
image2_anchor = mpimg.imread(image2_path_anchor)
plt.imshow(image2_anchor, cmap='gray')
plt.axis('off')
plt.title('Image 2')
plt.show()









import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP

# Specify the path to the CSV file
csv_file_path = "/home/j.maragall/j.maragall/Replication2 _DAPImapping/features.csv"

# Load the data from the CSV file
features_df = pd.read_csv(csv_file_path)

# Specify the percentage threshold for dropping
zero_threshold = 0.97

# Create a mask of boolean values representing whether each value in the DataFrame is zero
is_zero = features_df == 0

# Calculate the percentage of zeros in each column
zero_percentage = is_zero.mean()

# Identify the columns which have more than 97% zeros
columns_to_drop = zero_percentage[zero_percentage > zero_threshold].index

# Drop those columns from the DataFrame
features_df = features_df.drop(columns=columns_to_drop)

# Print the updated DataFrame
features_df.head()









import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Specify the path to the CSV file
csv_file_path = "/home/j.maragall/j.maragall/Replication2 _DAPImapping/features.csv"

# Load the data from the CSV file
features_df = pd.read_csv(csv_file_path)

# Specify the percentage threshold for dropping
zero_threshold = 0.97

# Create a mask of boolean values representing whether each value in the DataFrame is zero
is_zero = features_df == 0

# Calculate the percentage of zeros in each column
zero_percentage = is_zero.mean()

# Identify the columns which have more than 97% zeros
columns_to_drop = zero_percentage[zero_percentage > zero_threshold].index

# Drop those columns from the DataFrame
features_df = features_df.drop(columns=columns_to_drop)

# Create a scaler object
scaler = MinMaxScaler()

# Apply the scaler to the DataFrame
features_df_scaled = pd.DataFrame(scaler.fit_transform(features_df), columns = features_df.columns)

# Print the updated DataFrame
features_df_scaled.head()

# Save the scaled DataFrame to a new CSV file
new_csv_file_path = "/home/j.maragall/test_scaled_features.csv"
features_df_scaled.to_csv(new_csv_file_path, index=False)

print("Scaled data saved to:", new_csv_file_path)









from skimage import io

# Loading the image
transformed_img3 = io.imread("/home/j.maragall/j.maragall/Replication2 _DAPImapping/transformed_image.tif")







import tiffslide as openslide
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imsave
import cv2
import imageio
from tifffile import imread
from matplotlib import cm
from skimage import img_as_ubyte
import matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tiffslide as openslide
from tifffile import imread
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy
from joblib import Parallel, delayed
import pandas as pd
from umap import UMAP


def position_channels(image): # Get the channel index based on the lowest value in the tuple and visualize the image based on where the channels actually are.

    min_value = min(image.shape)

    if min_value == image.shape[0]:
        position = 0
    elif min_value == image.shape[1]:
        position = 1
    else:
        position = 2
    return position

def single_channel(slide, channel_index): #stores the single channel of an image, independently from where the channel value is in the imag.shape tuple
    location = position_channels(slide)
    
    if location == 0:
        image = slide[channel_index]
    elif location == 1:
        image = slide[:, channel_index]
    else:
        image = slide[:, :, channel_index]
    return image
    








    def normalize_mask(image, target_dtype=np.uint8):
    normalized_image = cv2.normalize(image, None, 0, np.iinfo(target_dtype).max, cv2.NORM_MINMAX)
    return normalized_image.astype(target_dtype)



read_DAPI = openslide.open_slide("/home/j.maragall/Pipeline_CellMapping/R2_DAPI_cleaned_mask.tif")
DAPI_mask = normalize_mask(np.array(read_DAPI.get_thumbnail(read_DAPI.dimensions)), np.uint8)




def normalize_CODEX(image):
  """Normalizes an image to uint8.

  Args:
    image: The image to be normalized.

  Returns:
    A normalized image in uint8 format.
  """

  max_value = np.iinfo(image.dtype).max
  image = image / max_value
  image = image * 255
  image = image.astype(np.uint8)

  return image

read_CODEX = imread("/blue/pinaki.sarder/j.maragall/CellMapping/15-1.tif")
CODEX = normalize_CODEX(read_CODEX)









from PIL import Image

def save_channels(image, channel):
    """
    Save the channel based on the lowest value in the tuple.

    Args:
        image: The image to save.
    """

    min_value = min(image.shape)

    if min_value == image.shape[0]:
        # The channels are in the first dimension
        slide = image[channel]
    elif min_value == image.shape[1]:
        # The channels are in the second dimension
        slide = image[:, channel]
    else:
        # The channels are in the third dimension
        slide = image[:, :, channel]

    # Convert to Image and save
    im = Image.fromarray(slide)
    im.save(f'channel_{channel}.png')

# specify channels you want to save
channels_to_save = [0]

for channel in channels_to_save:
    save_channels(CODEX, channel)









from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from scipy.stats import skew, kurtosis
from skimage.util import img_as_ubyte
from skimage import img_as_float

# Optimized function to extract new features from a single object
def extract_new_features(region_label):
    object_mask = (labeled_image == region_label)

    # Extract the channel and calculate the features
    new_channel_features = np.zeros(7 * CODEX.shape[position_channels(CODEX)], dtype=np.float32)
    for channel_index in range(CODEX.shape[position_channels(CODEX)]):
        single_channel_img = single_channel(CODEX, channel_index) 
        masked_single_channel = single_channel_img * object_mask

        channel_data = masked_single_channel[object_mask]

        try:
            # New features
            contrast = np.ptp(channel_data)
            energy = np.sum(channel_data**2)
            rms = np.sqrt(np.mean(channel_data**2))
            smoothness = 1 - 1 / (1 + np.var(channel_data))
            uniformity = np.sum(channel_data**2) / (np.sqrt(np.mean(channel_data**2)) + np.finfo(float).eps)
            skewness = skew(channel_data)
            kurt = kurtosis(channel_data)

            index = channel_index * 7
            new_channel_features[index:index+7] = [contrast, energy, rms, smoothness, uniformity, skewness, kurt]

        except Exception as e:
            print(f"Failed to extract features for region {region_label}, channel {channel_index}. Error: {e}")

    # Return the new object features
    return new_channel_features

# Parallel extraction
num_channels = CODEX.shape[position_channels(CODEX)]
num_new_features_per_channel = 7  # contrast, energy, rms, smoothness, uniformity, skewness, kurtosis

# Pre-allocate memory
new_features_array = np.zeros((number_of_objects, num_new_features_per_channel * num_channels), dtype=np.float32)

# Parallel extraction
new_features_list = Parallel(n_jobs=11)(delayed(extract_new_features)(region_label) for region_label in range(1, number_of_objects + 1))

# Copy the features from the list to the pre-allocated NumPy array
for i, features in enumerate(new_features_list):
    new_features_array[i, :] = features

# Convert to dataframe
new_columns = [f'channel_{i}_contrast' for i in range(num_channels)] + [f'channel_{i}_energy' for i in range(num_channels)] + \
          [f'channel_{i}_rms' for i in range(num_channels)] + [f'channel_{i}_smoothness' for i in range(num_channels)] + \
          [f'channel_{i}_uniformity' for i in range(num_channels)] + [f'channel_{i}_skewness' for i in range(num_channels)] + \
          [f'channel_{i}_kurtosis' for i in range(num_channels)] 

new_features_df = pd.DataFrame(new_features_array, columns=new_columns)

# Join the new features to the existing ones
final_df = pd.concat([features_df, new_features_df], axis=1)

# Save the final DataFrame to a new .csv file
final_df.to_csv('final_features.csv', index=False)









import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Specify the path to the CSV file
csv_file_path = "/home/j.maragall/final_features.csv"

# Load the data from the CSV file
features_df = pd.read_csv(csv_file_path)

# Impute all NaN values with 0
features_df.fillna(0, inplace=True)

# Create a scaler object
scaler = MinMaxScaler()

# Fit and transform the data
features_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)









# Specify the path to the CSV file
csv_file_path = "scaled_features.csv"

# Load the data from the CSV file
features_df = pd.read_csv(csv_file_path)







# Now let's extract features from each object in the binary mask
# First, label each object
labeled_image = label(DAPI_mask)
number_of_objects = labeled_image.max()
print(features_df.head())



# Convert list of arrays to dataframe
additional_features_df = pd.DataFrame(additional_features_list)

# Save the dataframe to a csv file
additional_features_df.to_csv('ADDIT_features.csv', index=False)









# The list of column names for the last seven columns
last_column_names = ['centroid_x', 'centroid_y', 'orientation', 'minr', 'minc', 'maxr', 'maxc']

# Get the current column names as a list
current_column_names = list(additional_features_df.columns)

# Replace the last seven column names
current_column_names[-7:] = last_column_names

# Assign the updated column names back to the DataFrame
additional_features_df.columns = current_column_names













pd.read_csv("/home/j.maragall/ADDIT_features.csv")


# Read the dataframe
df = pd.read_csv("/home/j.maragall/ADDIT_features.csv")

# Create a dictionary mapping old column names to new ones
rename_dict = {old: new for old, new in zip(df.columns, 
                sum([[f'channel_{i}_max_intensity', f'channel_{i}_min_intensity', f'channel_{i}_sum_intensity'] for i in range(44)], []) + \
                ['centroid_x', 'centroid_y', 'orientation', 'minr', 'minc', 'maxr', 'maxc'])}

# Use the dictionary to rename the columns
df.rename(columns=rename_dict, inplace=True)

# Save the DataFrame
df.to_csv('/home/j.maragall/ADDIT_renamed_features.csv', index=False)









import pandas as pd

# Load the dataframes
features_df = pd.read_csv("final_features_renamed.csv")
last_features_df = pd.read_csv('/home/j.maragall/ADDIT_renamed_features.csv')









import pandas as pd

# Load the dataframes
features_df = pd.read_csv("final_features_renamed.csv")
last_features_df = pd.read_csv('/home/j.maragall/ADDIT_renamed_features.csv')
last_features_df.head()








import pandas as pd

# Load the dataframes
features_df = pd.read_csv("final_features_renamed.csv")
last_features_df = pd.read_csv('/home/j.maragall/ADDIT_renamed_features.csv')


merged_df = pd.concat([features_df, last_features_df], axis=1)


merged_df.head()

merged_df.to_csv('/home/j.maragall/PRESENTATION_FEATURES.csv', index=False)



# Copy the features from the list to the pre-allocated NumPy array
for i, features in enumerate(additional_features_list):
    valid_features = np.concatenate([features[:132], features[220:]])  # Exclude columns 132-219
    additional_features_array[i, :] = valid_features

# Define column names
columns = ([f'channel_{i}_max_intensity' for i in range(44)] + 
           [f'channel_{i}_min_intensity' for i in range(44)] + 
           [f'channel_{i}_sum_intensity' for i in range(44)] + 
           ['centroid_x', 'centroid_y', 'orientation', 'minr', 'minc', 'maxr', 'maxc'])

# Convert to dataframe
additional_features_df = pd.DataFrame(additional_features_array, columns=columns)

# Save the additional features to a new .csv file
additional_features_df.to_csv('ADDIT_features.csv', index=False)





# Create a color map
cmap = matplotlib.colormaps.get_cmap('tab20')  # 'tab10' colormap has exactly 10 distinct colors

# Convert the labels to colors using the colormap and convert it to the desired type
colored_labels = cmap(labels / 20)[:, :3]  # Normalize the labels to [0, 1] for the colormap
colored_labels = img_as_ubyte(colored_labels)  # Convert to 8-bit unsigned byte format

# Create a new image where each pixel of an object is the color of the cluster it belongs to
colored_image = np.zeros((*DAPI_mask.shape, 3), dtype=np.uint8)  # Pre-allocate memory

# Iterate over all objects (assume that 'labels' is an array that contains the cluster id of each object)
for region_label in range(1, number_of_objects + 1):
    # Create a mask for the current object
    object_mask = (labeled_image == region_label)

    # Set the color of the object in the new image
    colored_image[object_mask] = colored_labels[labels[region_label - 1]]

# Now 'colored_image' is a color image where each object is colored according to its cluster

# Combine the images (you may want to adjust the alpha values to get the desired look)
combined_img = cv2.addWeighted(transformed_img3, 0.6, colored_image, 0.4, 0)

# Plotting the combined image
plt.imshow(combined_img)
plt.title("Overlay of Clustered Objects on Transformed Image")
plt.show()










def visualize_channels_with_overlay(image, cluster_image, channel):
    """
    Get the channel index based on the lowest value in the tuple and visualize the image based on where the channels actually are.
    Adds an overlay of the identified clusters to the visualized channel.

    Args:
        image: The image to visualize.
        cluster_image: The image showing identified clusters.
        channel: The channel to visualize.
    """

    min_value = min(image.shape)

    if min_value == image.shape[0]:
        # The channels are in the first dimension
        slide = image[channel]
        slide = cv2.cvtColor(slide, cv2.COLOR_GRAY2RGB)
        combined_img = cv2.addWeighted(slide, 0.6, cluster_image, 0.4, 0)
        plt.imshow(combined_img)
        plt.title(f"Overlay of Clustered Objects on Channel {channel}")
        plt.show()
    elif min_value == image.shape[1]:
        # The channels are in the second dimension
        slide = image[:, channel]
        slide = cv2.cvtColor(slide, cv2.COLOR_GRAY2RGB)
        combined_img = cv2.addWeighted(slide, 0.6, cluster_image, 0.4, 0)
        plt.imshow(combined_img)
        plt.title(f"Overlay of Clustered Objects on Channel {channel}")
        plt.show()
    else:
        # The channels are in the third dimension
        slide = image[:, :, channel]
        slide = cv2.cvtColor(slide, cv2.COLOR_GRAY2RGB)
        combined_img = cv2.addWeighted(slide, 0.6, cluster_image, 0.4, 0)
        plt.imshow(combined_img)
        plt.title(f"Overlay of Clustered Objects on Channel {channel}")
        plt.show()

#visualize
visualize_channels_with_overlay(CODEX, colored_image, 6)

#...

features_df = pd.read_csv("PRESENTATION_FEATURES.csv")

features_df.head()

!pip install joblib

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Impute all NaN values with 0
features_df.fillna(0, inplace=True)

# Create a scaler object
scaler = MinMaxScaler()

# Fit and transform the data
features_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)





import tiffslide as openslide
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imsave
import cv2
import imageio
from tifffile import imread

from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy
from joblib import Parallel, delayed
import pandas as pd
from umap import UMAP

features_df.shape

# Perform dimensionality reduction using UMAP
umap_model = UMAP(n_neighbors=7, min_dist=0.07, n_components=2, metric="canberra")
embedding = umap_model.fit_transform(features_df)

# Plot the results
plt.scatter(embedding[:, 0], embedding[:, 1], s=1)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP projection of the feature vectors')

# Print the parameters
print("UMAP Parameters:")
print(f"n_neighbors: {umap_model.n_neighbors}")
print(f"min_dist: {umap_model.min_dist}")
print(f"n_components: {umap_model.n_components}")
print(f"metric: {umap_model.metric}")

plt.show()


from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Apply the AgglomerativeClustering algorithm to the UMAP embeddings
agg_clustering = AgglomerativeClustering(n_clusters=9)  # adjust the number of clusters as needed
labels = agg_clustering.fit_predict(embedding)

# Calculate cluster centroids
centroids = np.array([embedding[labels == i].mean(axis=0) for i in range(9)])

# Plot the results with different colors for each cluster
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=1)

# Annotate the clusters
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], str(i), fontsize=12, ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP projection of the feature vectors with Hierarchical Clustering')
plt.colorbar(label='Cluster Label')
plt.show()

# Add cluster_labels column to the DataFrame
features_df['cluster_labels'] = labels


# Get list of all channels
channels = [col for col in features_df.columns if 'channel_' in col and '_mean' in col]

intensity_threshold = .7

for channel in channels:
    # Create a mask for nuclei over the intensity threshold in the current channel
    mask = features_df[channel] > intensity_threshold

    # Apply the mask to isolate the relevant nuclei
    filtered_df = features_df[mask]

    # Analyze the distribution of cluster labels within this group
    cluster_distribution = filtered_df['cluster_labels'].value_counts().sort_index()

    # Print or plot the distribution
    print(f"\nCluster distribution for nuclei with high {channel}:")
    print(cluster_distribution)
    
    if not cluster_distribution.empty:  # Check if the series is not empty
        cluster_distribution.plot(kind='bar', title=f'Cluster distribution for nuclei with high {channel}')
        plt.show()
    else:
        print(f"No nuclei exceeded the intensity threshold for {channel}")


# Group by 'cluster_labels', calculate variance for each feature
cluster_variances = features_df.groupby('cluster_labels').var()

# Identify top 5 features with highest variance for each cluster
for cluster in cluster_variances.index:
    top_features = cluster_variances.loc[cluster].nlargest(44)
    print(f"Top 44 features driving intra-cluster differences in cluster {cluster} are {top_features.index.tolist()}")
    
    # Calculate proportion of total variance explained by these features
    total_variance = cluster_variances.loc[cluster].sum()
    explained_variance = top_features.sum()
    print(f"These features explain {explained_variance / total_variance * 100:.2f}% of the total intra-cluster variance.")
    
    # Levene's test for these features across clusters
    from scipy.stats import levene
    for feature in top_features.index:
        stat, p = levene(*[features_df.loc[features_df['cluster_labels']==label, feature] for label in set(labels)])
        print(f"Levene Test for {feature} across clusters: W={stat}, p={p}")
    
# Save DataFrame to CSV
results_df.to_csv('results.csv', index=False)



pd.read_csv('results.csv')


# Create a new column in the dataframe to denote whether a data point is in cluster 5 or not
features_df['in_cluster_6'] = features_df['cluster_labels'] == 6

# Fit the ANOVA model
model = ols('channel_34_mean ~ C(in_cluster_6)', data=features_df).fit()

# Perform the ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Calculate η²
anova_table['eta_sq'] = anova_table[:-1]['sum_sq'] / sum(anova_table['sum_sq'])

print(anova_table)


from statsmodels.formula.api import ols
import statsmodels.api as sm

# Create new columns in the dataframe to denote whether a data point is in cluster 2 or 10
features_df['in_cluster_2'] = features_df['cluster_labels'] == 2
features_df['in_cluster_10'] = features_df['cluster_labels'] == 10

# Fit the ANOVA model for cluster 2
model_2 = ols('channel_30_mean ~ C(in_cluster_2)', data=features_df).fit()
# Perform the ANOVA
anova_table_2 = sm.stats.anova_lm(model_2, typ=2)
# Calculate η² for cluster 2
anova_table_2['eta_sq'] = anova_table_2[:-1]['sum_sq'] / sum(anova_table_2['sum_sq'])

# Fit the ANOVA model for cluster 10
model_10 = ols('channel_30_mean ~ C(in_cluster_10)', data=features_df).fit()
# Perform the ANOVA
anova_table_10 = sm.stats.anova_lm(model_10, typ=2)
# Calculate η² for cluster 10
anova_table_10['eta_sq'] = anova_table_10[:-1]['sum_sq'] / sum(anova_table_10['sum_sq'])

print("ANOVA table for cluster 2:")
print(anova_table_2)
print("\nANOVA table for cluster 10:")
print(anova_table_10)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_intra_cluster_variability(df, cluster_label):
    # Filter DataFrame for the specific cluster
    cluster_df = df.loc[df['cluster_labels'] == cluster_label]

    # Compute and print descriptive statistics
    descriptive_stats = cluster_df.describe()
    print("Descriptive Statistics:")
    print(descriptive_stats)
    
    # Compute and print correlations
    correlations = cluster_df.corr()
    print("\nCorrelations:")
    print(correlations)

    # Create a pairplot
    sns.pairplot(cluster_df)
    plt.show()

analyze_intra_cluster_variability(features_df, 3)
