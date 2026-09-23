#import modules
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as tff

#---------- Filters out faces in WIDER_FACE which are too difficult for the basic model ----------

def filter_annotations(boxes, scale_x, scale_y, min_size =8, max_blur=3, max_occlusion=4, max_pose=3):

        #stores good boxes
        good_boxes = []

        for annotation in boxes:

            x, y, width, height = annotation["bbox"]

            width = scale_x * width
            height = scale_y * height

            #removes small faces
            if width < min_size or height < min_size:
                continue

            #removes blurred faces
            if annotation["blur"] >= max_blur:
                continue

            #removes heavily occluded faces
            #in computer vision, it refers to things blocking the faces
            if annotation["occlusion"] >= max_occlusion:
                continue

            #removes invalid annotations
            if annotation["invalid"] != 0:
                continue

            #removes difficult poses
            if annotation["pose"] >= max_pose:
                continue

            good_boxes.append(annotation)

        #return filtered list of annotations
        removed = len(boxes) - len(good_boxes)
        return good_boxes, removed

#---------- Dataset class for converting an image + JSON labels into trainable tensors ----------

class FaceAsTensorDataset(Dataset):

    #image size is 224x224
    #images and labels stored in separate directories (image_dir and label_dir respectively)
    #they have the same filename but different extensions (e.g. image_1.jpg and image_1.json)
    def __init__(self, image_dir, label_dir, input_size=224):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.input_size = input_size
        self.small_grid_size = input_size // 8
        self.medium_grid_size = input_size // 16
        self.large_grid_size = input_size // 32

        #finds all image paths and sorts them to ensure they are in the same order as the labels
        self.images = sorted(
            p for p in self.image_dir.rglob("*") #use rglob to find files in subdirectories
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"] #only include image files with these extensions
        )

    #tells pytorch how many images are in the dataset, so it knows how many times to call __getitem__ during training
    def __len__(self):
            return len(self.images)

    #when this class is called, this function will be executed, and it will return a tuple of (image, target) tensors for the image at the given index
    def __getitem__(self, index):

        #finds corresponding image and label paths for the given index
        image_path = self.images[index]

        #adds in the parent directory name to the label path, since 
        #the labels are stored in subdirectories that match the image subdirectories
        label_path = (
        self.label_dir /
        f"{image_path.parent.stem}/{image_path.stem}.json"
        )

        #loads image and converts it to 3-channel RGB, loads JSON labels into a list of bounding boxes
        image = Image.open(image_path).convert("RGB")

        try:
            with open(label_path) as f:
                boxes = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            
            boxes = []


        #store original image dimensions
        original_width, original_height = image.size

        total_removed = 0
        total = 0

        #resize image to 128x128 while maintaining aspect ratio, and pad with black if necessary
        #prevents distorted faces
        image.thumbnail((self.input_size, self.input_size))

        #actual dimensions after PIL thumbnail()
        new_width = image.width
        new_height = image.height


        #derive the actual scale from the resized image
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        #call the filter function to return easier targets
        total = len(boxes)
        boxes, removed = filter_annotations(boxes, scale_x, scale_y)
        total_removed += removed


        #create an RGB canvas
        canvas = Image.new("RGB", (self.input_size, self.input_size), (0, 0, 0))  #black padding

        #center the image
        x = (self.input_size - image.width) // 2
        y = (self.input_size - image.height) // 2

        canvas.paste(image, (x, y))

        #convert image to tensor
        image = tff.to_tensor(canvas)


        #normalize tensors to have values between -1 and 1, as opposed to 0 and 1
        #this is standard for CNN training
        image = tff.normalize(
            image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )


        #create a target tensor of shape (grid_size, grid_size, 5)
        #last dimension corresponds to [confidence score, x, y, width, height] of the detected face bounding box
        #this is the same format as the output of the FaceDetector model, so it can be used to calculate the loss during training
        small_target = torch.zeros(
            self.small_grid_size,
            self.small_grid_size,
            5
        )

        medium_target = torch.zeros(
            self.medium_grid_size,
            self.medium_grid_size,
            5
        )

        large_target = torch.zeros(
            self.large_grid_size,
            self.large_grid_size,
            5
        )

        #loops through each bounding box in the JSON labels
        #extra information is ignored, as we only care about the bounding box coordinates
        if len(boxes) !=0:
            for box in boxes:

                #extracts annotation data in WIDER FACE format: [x, y, width, height]
                x1, y1, box_width, box_height = box["bbox"]

                #ignore invalid annotations
                if box_width <= 0 or box_height <= 0:
                    continue

                #set other corner coordinates
                x2 = x1 + box_width
                y2 = y1 + box_height

                #padding
                pad_x = (self.input_size - new_width) // 2
                pad_y = (self.input_size - new_height) // 2

                #transform bounding box
                x1 = x1 * scale_x + pad_x
                y1 = y1 * scale_y + pad_y
                x2 = x2 * scale_x + pad_x
                y2 = y2 * scale_y + pad_y

                #centre
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                #calculate width / height of bounding box
                width = x2 - x1
                height = y2 - y1

                #make sure widths/heights are not negative- this corrupted my training datas
                assert width > 0, f"Negative/zero width: {box['bbox']}"
                assert height > 0, f"Negative/zero height: {box['bbox']}"

                #normalise to [0, 1]
                cx_norm = cx / self.input_size
                cy_norm = cy / self.input_size
                w_norm = width / self.input_size
                h_norm = height / self.input_size

                if not (0 <= cx_norm <= 1 and 0 <= cy_norm <= 1):
                    continue

                #face size is maximum length of box (might change to area)
                face_size = max(width, height)

                #assigns faces to the correct sized target and adjusts grid size
                if face_size < 32:
                    target = small_target
                    grid_size = self.small_grid_size
                elif face_size < 64:
                    target = medium_target
                    grid_size = self.medium_grid_size

                else:
                    target = large_target
                    grid_size = self.large_grid_size

                #calculate the grid cell
                cell_x = cx_norm * grid_size
                cell_y = cy_norm * grid_size

                #grid cells go from 0-15, so we have to clamp at 15 to prevent cell 16 being accessed
                grid_x = min(int(cell_x), grid_size - 1)
                grid_y = min(int(cell_y), grid_size - 1)

                #we use relative coordinates, because these values make regression easier
                #the model is able to learn more effectively
                x_offset = cell_x - grid_x
                y_offset = cell_y - grid_y

                #assign target
                if target[grid_y, grid_x, 0] == 0:
                    target[grid_y, grid_x, 0] = 1.0
                    target[grid_y, grid_x, 1] = x_offset
                    target[grid_y, grid_x, 2] = y_offset  
                    target[grid_y, grid_x, 3] = w_norm
                    target[grid_y, grid_x, 4] = h_norm

        #same processing as model uses- flattens predictions into 1D list 
        small_target = small_target.reshape(-1, 5)
        medium_target = medium_target.reshape(-1, 5)
        large_target = large_target.reshape(-1, 5)

        return image, small_target, medium_target, large_target, total, total_removed