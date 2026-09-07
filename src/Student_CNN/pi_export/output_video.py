import cv2
import torch
import numpy as np
import torchvision.transforms as tf
from pathlib import Path
from Student_CNN.student_model.student import FaceDetector
from torchvision.ops import nms

#---------- Letterbox function, matches trainign data ----------

def letterbox_image(image, target_size=128):

    #image.shape outputs height, width, no. channels
    image_height, image_width, _ = image.shape

    #calculates the most dramatic scaling required
    scale = min(target_size / image_height, target_size / image_width)

    #applies scaling to both height and width
    new_height = int(image_height * scale)
    new_width = int(image_width * scale)

    #resizes image and preserves aspect ratio
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    #creates an array of zeros in the correct image dimensions
    #this means padding will be black
    new_img = np.zeros(
        (target_size, target_size, 3),
        dtype=np.uint8
    )

    #caluclates the amount of padding required for x and y dimensions of image
    #either one of pad_x or pad_y will be zero
    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2

    #fills array with the resized image 
    new_img[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    return new_img, scale, pad_x, pad_y, image_height, image_width

#---------- Load model from file ----------

def load_model(model_path):

    #takes model path input and loads the pytorch checkpoint
    weights = torch.load(model_path)
    model = FaceDetector()
    model.load_state_dict(weights["model_state_dict"])
    model.eval()
    return model

#---------- Decode model outputs into bounding box coordinates ----------

def decode_predictions(prediction, confidence_threshold=0.8, grid_size=16, image_size=128, iou_threshold=0.3):

    #sanity checks for different inputs:

    #check if single image, and store in a variable
    single_image = prediction.dim() == 3

    #if single image, add batch dimension
    if single_image:
        prediction = prediction.unsqueeze(0)

    #if neither single or batched image, raise an error
    elif prediction.dim() != 4:
        raise ValueError(
            f"Expected [H,W,5] or [B,H,W,5], "
            f"got {prediction.shape}"
        )

    #store the device the predictions are on
    device = prediction.device

    #create grid to apply coordinates to
    gy, gx = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij"
    )

    #convert from torch tensors to floats
    gx = gx.float()
    gy = gy.float()

    #flattens grid, turns 2d gx and gy into 1d 16x16 grid
    gx = gx.reshape(-1)
    gy = gy.reshape(-1)


    #[B, H, W, 5] -> [B, H*W, 5]
    #flattens predictions, H*W = cell no.
    prediction = prediction.reshape(
        prediction.size(0),
        -1,
        5
    )


    #decode confidence probability into confidence value
    confidence = torch.sigmoid(prediction[..., 0])

    #decode offset probability into relative offset
    x_offset = torch.sigmoid(prediction[..., 1])
    y_offset = torch.sigmoid(prediction[..., 2])

    #decode width/height probability into values relative to image
    width = torch.sigmoid(prediction[..., 3]) * image_size
    height = torch.sigmoid(prediction[..., 4]) * image_size

    cx = (
        (gx.unsqueeze(0) + x_offset)
        / grid_size
        * image_size
    )

    cy = (
        (gy.unsqueeze(0) + y_offset)
        / grid_size
        * image_size
    )

    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2

    #creates a tensor containing box tensors
    boxes = torch.stack(
        [x1, y1, x2, y2],
        dim=-1
    )

    #initialise empty lists for batch information
    batch_boxes = []
    batch_scores = []

    #iterate over boxes for an image at a time
    for b in range(prediction.size(0)):

        #keep a predicition if confidence greater than threshold
        keep = confidence[b] >= confidence_threshold

        boxes_b = boxes[b][keep]
        scores_b = confidence[b][keep]

        #if there are no predicitions with high enough confidence, return an empty tensor 
        #for the scores and boxes
        if boxes_b.numel() == 0:
            batch_boxes.append(
                torch.empty(
                    (0, 4),
                    device=device
                )
            )

            batch_scores.append(
                torch.empty(
                    (0,),
                    device=device
                )
            )

            continue

        #if images overlap more than the iou_threshold, they are not included
        keep_nms = nms(
            boxes_b,
            scores_b,
            iou_threshold
        )

        #only keep images that do not overlap
        #boxes don't represent same face
        batch_boxes.append(boxes_b[keep_nms])
        batch_scores.append(scores_b[keep_nms])

    #output only one image if single image input
    if single_image:
        return batch_boxes[0], batch_scores[0]

    return batch_boxes, batch_scores


def main():

    #image size
    image_size = 128

    #find package root
    package_root = Path(__file__).parent.parent

    model = load_model(package_root / "student_checkpoint.pt")

    #checks if CUDA-compatible GPU is available for efficiency, otherwise uses CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        ret, frame = cap.read()

        #check frame was read correctly
        if not ret:
            print("Could not read frame")
            break

        original_frame = frame

        #OpenCV gives BGR; convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #letterbox image, so the model can process it accurately
        rgb, scale, pad_x, pad_y, original_height, original_width = letterbox_image(rgb, 128)

        #convert image to tensor
        tensorImg = tf.ToTensor()(rgb)    

        #for reliable results, we copy training normalization values from the FaceAsTensorDataset class in dataset.py
        normalize = tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        tensorImg = normalize(tensorImg)

        #add a batch dimension, since the model expects this as input
        test_image_tensor = tensorImg.unsqueeze(0)

        test_image_tensor = test_image_tensor.to(device)

        #run model, making sure we don't compute gradients for efficiency
        with torch.no_grad():

            output = model(test_image_tensor)

        #moves prediction back to CPU
        output = output[0].cpu()

        #draws detections
        boxes, scores = decode_predictions(output)

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = box.tolist()

            #transforms coordinates back into scale of original image
            x1 = (x1 - pad_x) / scale
            x2 = (x2 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            y2 = (y2 - pad_y) / scale

            #clamps to original image
            x1 = max(0, min(original_width - 1, x1))
            x2 = max(0, min(original_width - 1, x2))
            y1 = max(0, min(original_height - 1, y1))
            y2 = max(0, min(original_height - 1, y2))

            #converts coordinates to integer pixel values, which are required for openCV functions
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            #openCV function to draw a rectangular bounding box
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color=(255,0 ,0))

            #font for confidence score label
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = f"Face: {score.item():.2f}"

            #draws confidence score label on corner of bounding box
            cv2.putText(original_frame, label, (x1, y1), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        #displays frame in window
        cv2.imshow("Face Detection", original_frame)

        #press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #cleanup after window is closed
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()