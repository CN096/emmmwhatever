import cv2
import numpy as np

def compute_integral_image(image):
    """
    Computes the integral image of the input image.
    """
    return cv2.integral(image)

def compute_haar_feature(integral_image, x, y, w, h):
    """
    Computes the value of an upright Haar feature (top-black, bottom-white).
    """
    # Top rectangle (black)
    top_sum = integral_image[y + h // 2, x + w] - integral_image[y + h // 2, x] - \
              integral_image[y, x + w] + integral_image[y, x]

    # Bottom rectangle (white)
    bottom_sum = integral_image[y + h, x + w] - integral_image[y + h, x] - \
                 integral_image[y + h // 2, x + w] + integral_image[y + h // 2, x]

    # Haar feature value
    return bottom_sum - top_sum

def sliding_window_detection(integral_image, feature_size, step_size, threshold):
    """
    Detect regions with a Haar feature exceeding a threshold using a sliding window.
    """
    detected_regions = []
    image_height, image_width = integral_image.shape[:2]

    # Adjust y range and x range to ensure full coverage
    for y in range(0, image_height - feature_size[1] + 1, step_size):
        for x in range(0, image_width - feature_size[0] + 1, step_size):
            haar_value = compute_haar_feature(integral_image, x, y, feature_size[0], feature_size[1])
            if haar_value > threshold:
                detected_regions.append((x, y, feature_size[0], feature_size[1]))

    return detected_regions

def non_maximum_suppression(boxes, overlap_thresh):
    """
    Perform non-maximum suppression on the detected regions.
    """
    if len(boxes) == 0:
        return []

    # Convert to numpy array
    boxes = np.array(boxes)

    # Coordinates of the boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Area of the boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by the bottom-right y-coordinate of the bounding box
    order = np.argsort(y2)

    picked = []

    while len(order) > 0:
        i = order[-1]
        picked.append(i)

        # Compute overlap
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[order[:-1]]

        order = order[np.where(overlap <= overlap_thresh)[0]]

    return boxes[picked].tolist()

def main(image_path, feature_size, step_size, threshold, overlap_thresh):
    """
    Main function to perform face detection using Haar features.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert to grayscale for processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute integral image
    integral_image = compute_integral_image(gray_image)

    # Detect regions using sliding window
    detected_regions = sliding_window_detection(
        integral_image, feature_size, step_size, threshold
    )

    # Apply non-maximum suppression
    final_regions = non_maximum_suppression(detected_regions, overlap_thresh)

    # Display results
    result_image = image.copy()  # Avoid altering the original image
    for (x, y, w, h) in final_regions:
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Resize the output window to show the entire image
    resized_image = cv2.resize(result_image, (1000, int(1000 * result_image.shape[0] / result_image.shape[1])))
    cv2.imshow("Detected Faces", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parameters
    image_path = "D:\\Users\\CN096\\computervision\\computervision\\face.jpg"  # Updated path to uploaded image
    feature_size = (600, 600)       # Adjusted size of Haar feature (width, height)
    step_size = 15                # Sliding window step size
    threshold = 11919810           # Adjusted threshold for Haar feature value
    overlap_thresh = 0.1         # Overlap threshold for NMS

    main(image_path, feature_size, step_size, threshold, overlap_thresh)