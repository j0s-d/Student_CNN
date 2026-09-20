from pathlib import Path
import torch 
import cv2
import torchvision.transforms as tf
from Student_CNN.pi_export.output_video import decode_predictions, letterbox_image, apply_nms, load_model

def main():

    #checks if CUDA-compatible GPU is available for efficiency, otherwise uses CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using Device: {device}")

    #image size
    image_size = 224

    #find package root
    package_root = Path(__file__).parent.parent

    model = load_model(package_root, student=True)
    model.to(device)

    #opens image
    image_file = package_root / "test_images" / "soldier1.jpg"
    test_image = cv2.imread(image_file, cv2.IMREAD_COLOR)

    if test_image is None:
        raise FileNotFoundError("Image not found")

    #OpenCV gives BGR; convert to RGB
    rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
    #stores output location
    output_file = package_root / "test_images" / "output_soldier1.png"

    #copy original image
    original_frame = test_image.copy()

    #letterbox image, so the model can process it accurately
    rgb, scale, pad_x, pad_y, original_height, original_width = letterbox_image(rgb, image_size)

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

        small_faces, medium_faces, large_faces = model(test_image_tensor)


    #moves prediction back to CPU
    small_faces = small_faces[0].cpu()
    medium_faces = medium_faces[0].cpu()
    large_faces = large_faces[0].cpu()

    small_faces, small_scores = decode_predictions(small_faces, image_size // 8)
    medium_faces, medium_scores = decode_predictions(medium_faces, image_size // 16)
    large_faces, large_scores = decode_predictions(large_faces, image_size // 32)
    boxes = torch.cat([small_faces, medium_faces, large_faces], dim=0)
    scores = torch.cat([small_scores, medium_scores, large_scores], dim=0)

    #apply nms to boxes
    boxes, scores = apply_nms(boxes, scores)

    #draws detections
    for box, score in zip(boxes, scores):
        
        x1, y1, x2, y2 = box.tolist()

        #convert coordinates to pixels
        x1 *= image_size
        x2 *= image_size
        y1 *= image_size
        y2 *= image_size

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
        cv2.rectangle(original_frame, (x1, y1), (x2, y2), color=(0,0,255))

        #font for confidence score label
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"Face: {score.item():.2f}"

        #draws confidence score label on corner of bounding box
        cv2.putText(original_frame, label, (x1, y1), font, 0.5,(255,255,255),1,cv2.LINE_AA)


    #saves image
    print(f"Saving image to .... {output_file}")
    cv2.imwrite(output_file, original_frame)


if __name__ == "__main__":
    main()