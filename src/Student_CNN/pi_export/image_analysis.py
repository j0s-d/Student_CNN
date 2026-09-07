from pathlib import Path
import torch 
import torchvision.transforms.functional as tff
from PIL import Image, ImageDraw
from Student_CNN.teacher_model.teacher import TeacherFaceDetector
from Student_CNN.pi_export.output_video import decode_predictions

def load_model(model_path, DEVICE=torch.device('cpu')):
    #load model from file

    weights = torch.load(model_path)
    model = TeacherFaceDetector()
    model.load_state_dict(weights["model_state_dict"])
    model.eval()
    return model


def main():

    #checks if CUDA-compatible GPU is available for efficiency, otherwise uses CPU
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using Device: {DEVICE}")

    #find package root
    package_root = Path(__file__).parent.parent

    model = load_model(package_root / "teacher_checkpoint.pt", DEVICE=DEVICE)

    try:
        #opens image and converts to 3-channel RGB
        test_image = Image.open(package_root / "test_images" / "overfit1.jpg").convert("RGB")
    except:
        print("Image not found")
        

    #stores output location
    output_file = package_root / "test_images" / "overfit1_output.png"

    #store original image dimensions
    original_size = test_image.size

    #set variable for model image size
    model_size = 128

    #resize image to 128x128 while maintaining aspect ratio, and pad with black if necessary
    #prevents distorted faces
    image = test_image.copy()
    image.thumbnail((model_size, model_size))

    #create a 128x128 RGB canvas
    canvas = Image.new("RGB", (model_size, model_size), (0, 0, 0))  #black padding

    #center the image
    x = (model_size - image.width) // 2
    y = (model_size - image.height) // 2

    canvas.paste(image, (x, y))

    #convert image to tensor
    image = tff.to_tensor(canvas)

    #for reliable results, we copy training normalization values from the FaceAsTensorDataset class in dataset.py
    tff.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    #add a batch dimension, since the model expects this as input
    test_image_tensor = image.unsqueeze(0)

    #run model on image
    with torch.no_grad():

        #returns with boxes and scores from image
        boxes, scores = decode_predictions(model(test_image_tensor))

    #creates a drawing object in pillow which can be used to draw 
    draw = ImageDraw.Draw(test_image)

    #calculates scale for images
    scale_x = original_size[0] / model_size
    scale_y = original_size[1] / model_size

    for box, score in zip(boxes[0], scores[0]):
        
        #extracts the corners from the box list
        x1, y1, x2, y2 = box.tolist()

        #scales thumbnail coordinates to original image coordinates
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        #draws bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red",width=3)

        #writes confidence score on a red background
        draw.text((x1, max(0, y1 - 15)), f"{score.item():.2f}", fill="red")

    #saves image
    print(f"Saving image to .... {output_file}")
    test_image.save(output_file)


if __name__ == "__main__":
    main()