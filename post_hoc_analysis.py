from captum.attr import GuidedGradCam
from captum.attr import visualization as viz
import numpy as np
import cv2 as cv

def format_img(input_img):
    """
    Formats the input image by ensuring it has 4 dimensions.
    
    Parameters:
    - input_img (torch.Tensor): Input image tensor.
    
    Returns:
    - torch.Tensor: Formatted image tensor with 4 dimensions.
    """
    if len(input_img.shape) < 4:
        return input_img.unsqueeze(0)
    return input_img

def guidedgradcam(model, input_img, original_img, target_class):
    """
    Applies GuidedGradCam on the given model and image.
    
    Parameters:
    - model (torch.nn.Module): Model to apply GuidedGradCam on.
    - input_img (torch.Tensor): Input image tensor.
    - target_img (torch.Tensor): Target image tensor.
    - target_class (int): Target class for attribution.
    
    Returns:
    - tuple: Gradient visualizations and the original image.
    """
    # Ensure the images are correctly formatted
    input_img = format_img(input_img)
    original_img = format_img(original_img).squeeze(0)
    
    # Enable gradient computation for the input
    input_img.requires_grad = True
    
    # Initialize GuidedGradCam and compute the gradients
    guided_gc = GuidedGradCam(model, model.features[10])
    grads = guided_gc.attribute(input_img, target=target_class)
    
    # Transform the gradients and original image for visualization
    grads = np.transpose(grads.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    original_image = np.transpose((original_img.cpu().detach().numpy()), (1, 2, 0))
    # original_image = None
    return grads,original_image

def get_attention_maps(model, image, flare_probs):
    """
    Performs post-hoc analysis using GuidedGradCam.
    
    Parameters:
    - model (torch.nn.Module): Model for the analysis.
    - image (torch.Tensor): Image tensor for the analysis.
    - flare_probs (float): Flare probability.
    
    Returns:
    - tuple: Gradient visualizations and the original image.
    """
    # Determine the target class based on the flare probability
    target_class = 1 #if flare_probs >= 0.5 else 0
    
    # Apply GuidedGradCam
    guidedgradcam_grads ,original_image = guidedgradcam(model, image, image, target_class)
    # guidedgradcam_grads = (guidedgradcam_grads - guidedgradcam_grads.min()) / (guidedgradcam_grads.max() - guidedgradcam_grads.min())

    
    return guidedgradcam_grads,original_image


def superimpose_original(original_map, attention_map,alpha=0.45):

    original_map *= 255 

    # Load the 1-channel images from .npy files
    original_map = original_map.astype(int).squeeze()  # Squeeze the last dimension
    attention_map = attention_map.squeeze()

    # Normalize the attention map to [0, 1]
    normalized_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Apply the 'jet' colormap to the normalized attention map
    colormap = cv.applyColorMap((normalized_map * 255).astype(np.uint8), cv.COLORMAP_JET)

    # Ensure both images have the same number of channels
    if original_map.ndim == 2:
        original_map = cv.cvtColor(original_map.astype(np.uint8), cv.COLOR_GRAY2BGR)  # Convert grayscale to RGB

    # Blend the attention map with the original map
    # alpha = 0.45
    blended_image = cv.addWeighted(original_map, 1 - alpha, colormap, alpha, 0)

    # Normalize the final blended image from 0-1 and convert it to RGB for visualization
    final = cv.cvtColor(blended_image, cv.COLOR_BGR2RGB)
    final = (final - final.min()) / (final.max() - final.min())
    return final


def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())


def superimpose_image(original_map, attention_map, alpha=0.45):
    original_map *= 255

    # Load the 1-channel images from .npy files
    original_map = original_map.astype(int).squeeze()  # Squeeze the last dimension
    attention_map = attention_map.squeeze()

    # Normalize the attention map to [0, 1]
    normalized_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Convert the normalized attention map to a grayscale image
    grayscale_map = (normalized_map * 255).astype(np.uint8)

    # Ensure the original map is in grayscale
    if original_map.ndim == 2:
        original_map = original_map.astype(np.uint8)
    else:
        original_map = cv.cvtColor(original_map.astype(np.uint8), cv.COLOR_BGR2GRAY)

    # Blend the attention map with the original map
    blended_image = cv.addWeighted(original_map, 1 - alpha, grayscale_map, alpha, 0)

    # Normalize the final blended image
    final = (blended_image - blended_image.min()) / (blended_image.max() - blended_image.min())
    return final


def superimpose_circular_edge_npy(npy_path_original, background_image):
    """
    This function takes paths to two .npy files: one with a circular object and another background image.
    It finds the circular edge in the first image, resizes the second image to match the first one,
    and superimposes the circular edge onto the second image, blacking out the rest.
    
    :param npy_path_original: The path to the .npy file of the original image with the circular object.
    :param npy_path_background: The path to the .npy file of the background image to superimpose the edge onto.
    :param output_path: The path where the output image will be saved.
    :return: The path to the saved image with the superimposed edge.
    """
    import cv2
    import numpy as np
    
    # Load the .npy files to get the images as NumPy arrays
    original_image = np.load(npy_path_original)
    # background_image = np.load(npy_path_background)

    # Ensure the images are in the correct format (8-bit grayscale)
    if original_image.dtype != np.uint8:
        original_image = (original_image / original_image.max() * 255).astype(np.uint8)
    if background_image.dtype != np.uint8:
        background_image = (background_image / background_image.max() * 255).astype(np.uint8)
    
    # Find the largest contour in the original image
    _, thresholded_image = cv2.threshold(original_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Resize the background image to match the original image's dimensions
    resized_background_image = cv2.resize(background_image, (original_image.shape[1], original_image.shape[0]))
    
    # Create a mask for the largest contour
    mask = np.zeros_like(resized_background_image, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    
    # Superimpose the contour on the resized background image
    superimposed_image = cv2.bitwise_and(resized_background_image, resized_background_image, mask=mask)
    
    # Draw the circular edge in white on the superimposed image
    cv2.drawContours(superimposed_image, [largest_contour], -1, (255), thickness=1)
        
    return superimposed_image


def annotate_points(df,flares,img):

    radius = 5

    within = df[df.Distance == 0]['Flare'].to_list()

    if flares is not None:
        for fl in flares:
            point =  fl[1]
            flare = fl[0]
            # print(f"{flare} - {point}")

            if flare in within:
                cv.circle(img, point, radius, (255), -1)

            else:
                cv.drawMarker(img, point, (255), markerType=cv.MARKER_CROSS, markerSize=radius*2+1, thickness=2)

            # Add flare text near the flare point
            cv.putText(img, str(fl[0]), (point[0] + radius, point[1] - radius), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv.LINE_AA)
    return img