#import modules
import json
from pathlib import Path
import shutil

#-------- Function to check if the line is an image path (used when debugging)--------

def is_image_path(line):

    line = line.strip().lower()

    return (
        line.endswith(".jpg")
        or line.endswith(".jpeg")
        or line.endswith(".png")
    )

#-------- Function to convert WIDER FACE annotations to JSON format --------

def convert_wider_to_json(annotation_file, image_root, output_root):

    #set up paths
    annotation_file = Path(annotation_file)
    image_root = Path(image_root)
    output_root = Path(output_root)

    #create output directories for images and labels
    images_out = output_root / "images"
    labels_out = output_root / "labels"


    #read the entire file while it is open (prevents IO errors)
    with open(annotation_file, "r") as f:
        lines = [line.strip() for line in f]

    #set up counters for while loop
    i = 0
    image_count = 0

    #loop through the lines of the annotation file and retrieve the image path, number of faces, and annotations for each face
    while i < len(lines):

        #skip blank lines
        if not lines[i].strip():
            i += 1
            continue

        #stores the relative path of the image
        image_rel_path = lines[i]

        #use function to check if the line is an image path (used when debugging)
        if not is_image_path(image_rel_path):
            raise ValueError(
                f"Expected image path at line {i}, "
                f"but got: {image_rel_path!r}"
            )

        #increment line counter
        i += 1

        #store the number of faces in the image
        if i >= len(lines):
            break

        num_faces = int(lines[i])

        #checks number of faces is an integer
        #prints out debugging information if it is not an integer
        try:
            num_faces = int(num_faces)
        except ValueError:
            print("\n!!! PARSING ERROR !!!")
            print(f"image_rel_path = {image_rel_path!r}")
            print(f"num_faces = {num_faces!r}")
            print(f"line index = {i}")
            print(f"previous line = {lines[i-1]!r}")
            print(f"next line = {lines[i+1]!r}")
            raise

        #increment line counter
        i += 1

        #store the annotations for each face in the image
        annotations = []

        #accounts for WIDERFACE's extra dummy line for zero-face images
        if num_faces > 0:

            #loops over the number of faces in the image and retrieves the annotations for each face
            for _ in range(num_faces):

                values = list(map(int, lines[i].split()))
                i += 1

                if len(values) < 10:
                    raise ValueError(
                        f"Invalid annotation for "
                        f"{image_rel_path}: {values}"
                    )

                annotations.append({
                    "bbox": values[0:4],
                    "blur": values[4],
                    "expression": values[5],
                    "illumination": values[6],
                    "invalid": values[7],
                    "occlusion": values[8],
                    "pose": values[9]
                })

        else:
            if i < len(lines):

                dummy = lines[i]

                #checks if the dummy line is valid (contains 10 integers)
                parts = dummy.split()

                #increments line counter if the dummy line is valid (contains 10 integers)
                if len(parts) == 10:
                    try:
                        [int(x) for x in parts]
                        i += 1
                    except ValueError:
                        pass


        #create paths for the source image, output image, and output JSON file
        source_image = image_root / image_rel_path
        output_image = images_out / image_rel_path
        output_json = (labels_out / Path(image_rel_path).with_suffix(".json"))

        #makes directories for the output image and output JSON file if they do not already exist
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        #checks that source image exists before copying it to the output directory, raises an error if it does not exist
        if not source_image.exists():
            raise FileNotFoundError(
                f"Image does not exist:\n{source_image}"
            )

        else:

            shutil.copy2(
                source_image,
                output_image
            )


        #writes the annotations to the output JSON file
        with open(output_json, "w") as out:
            json.dump(
                annotations,
                out,
                indent=4
            )

        #increments the image count and prints out the number of images converted so far
        image_count += 1
        print(f"Converted #{image_count}: {image_rel_path}")



#training conversion of WIDER FACE annotations to JSON format
convert_wider_to_json(
    annotation_file = "",
    output_root = "",
    image_root= ""
)

#validation conversion of WIDER FACE annotations to JSON format
convert_wider_to_json(
    annotation_file = "",
    output_root = "",
    image_root= ""
)