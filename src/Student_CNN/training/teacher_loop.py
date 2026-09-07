#import modules
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from Student_CNN.teacher_model.teacher import TeacherFaceDetector
from . dataset import FaceAsTensorDataset
from . loss_function import detection_loss

#---------- Load checkpoint if it exists ----------

def LoadCheckpoint(model, optimizer, filename):

    #checks if checkpoint file exists
    if os.path.isfile(filename):

        print(f"Loading checkpoint '{filename}'")

        #loads each component in the correct order- very important
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        #print start epoch and checkpoint filename, to confirm that the checkpoint was loaded correctly
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")

        return start_epoch

    else:
        print(f"No checkpoint found at '{filename}'")
        return 0

#---------- Validation loop ----------

def Validation(model, val_loader, device="cpu"):

    #set model into evaluation mode, which affects the behaviour of the batchnorm2d block
    model.eval()

    #initialise variables
    validation_loss = 0.0
    positive_confidence = 0.0
    positive_count = 0.0

    #disables gradient calculation during validation
    #this saves GPU memory and computation
    with torch.no_grad():

        for images, targets in val_loader:

            #move validation data to device used for training
            images = images.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            #forward pass only through the model
            predictions = model(images)

            #calculate validation loss
            loss = detection_loss(
                predictions,
                targets
            )

            #tally validation loss
            validation_loss += loss.item()

            #get confidence for positive targets, applies mask to the predictions
            positive_mask = targets[..., 0] == 1
            confidence = torch.sigmoid(predictions[..., 0])[positive_mask]

            #tallies up how confident the model is in locations where faces exist
            if confidence.numel() > 0:
                positive_confidence += confidence.mean().item()
                positive_count += confidence.numel()


    #average validation loss across batches
    average_val_loss = validation_loss / len(val_loader)

    #average pos confidence
    average_pos_confidence = (
        positive_confidence / positive_count
        if positive_count > 0
        else 0.0
    )

    #print statistics
    print(
        f"validation_loss = {average_val_loss:.4f}"
        f"positive_confidence = {average_pos_confidence:.4f}"
    )

    return average_val_loss

#put all code in a main function, so that it can be called when the script is run directly 
#but not when it is imported as a module
def main():


    #find root directory for later
    root_directory = Path(__file__).parent.parent

    #EPOCHS is the number of times the model will see the entire dataset during training
    EPOCHS = 30

    #checks if CUDA-compatible GPU is available for efficiency, otherwise uses CPU
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    #check we are using GPU
    print(f"Using device: {DEVICE}")

    #creates a dataset instance
    #passes image/target tensors when the data loader is iterated over (__getitem__ is called)
    train_dataset = FaceAsTensorDataset(
        image_dir=root_directory / "data" / "WIDER_JSON" / "train" / "images",
        label_dir=root_directory / "data" / "WIDER_JSON" / "train" / "labels",
        input_size=128,
        grid_size=16
    )

    #creates a data loader for the dataset, loading 64 images at a time
    #shuffles the data for each epoch, so model does not learn the order of the data, which would reduce generalisation
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=12, #load data in parallel using 4 worker threads, recommended at 4 * no. GPU's
        
        ##allows data to be transferred to GPU through page-locked memory, which is faster than normal memory
        #CPU and GPU can access page-locked memory simultaneously, so data can be transferred to GPU while CPU is loading the next batch of data
        pin_memory=True, 

        #keeps workers running over each epoch
        persistent_workers=True
    )

    #same as the training dataset, but for validation data
    #used to evaluate the models performance on unseen data
    val_dataset = FaceAsTensorDataset(
        image_dir=root_directory / "data" / "WIDER_JSON" / "val" / "images",
        label_dir=root_directory / "data" / "WIDER_JSON" / "val" / "labels",
        input_size=128,
        grid_size=16
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        #keeps workers running over each epoch
        persistent_workers=True
    )

    #puts model onto device
    model = TeacherFaceDetector().to(DEVICE)

    #creates an AdamW optimiser for the model parameters
    #weight decay is a regularisation technique that reduces overfitting by penalising large weights
    #this prevents the model from memorising the training data, and encourages it to learn generalisable features
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    #creates a cosine annealing learning rate scheduler 
    #reduces the learning rate over time to help the model converge to a minimum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    #initialize start_epoch to 0, will be updated if checkpoint is loaded
    start_epoch = 0

    #load checkpoint file
    checkpoint_file = "teacher_checkpoint.pt"

    start_epoch = LoadCheckpoint(model, optimizer, filename=root_directory / checkpoint_file)


    #set best model parameters
    best_val_loss = float("inf")


    #---------- Main loop ----------

    for epoch in range(start_epoch, EPOCHS):

        #----------- Training loop ----------

        #print epoch number, to monitor training progress
        print(f"Epoch {epoch + 1}/{EPOCHS} ... Training ...")

        #puts model into training mode, which enables dropout and batch normalisation layers to behave differently during training and evaluation
        #for example, nn.BatchNorm2d uses the mean and variance of the current batch during training, but uses the running mean and variance during evaluation
        model.train()

        
        #update learning rate after each epoch, to help the model converge to a minimum
        scheduler.step(epoch)

        #set running loss and accuracy to 0 for each epoch, to calculate average at the end of the epoch
        running_loss = 0.0

        #initialize batch counter, to monitor training progress
        current_batch = 0

        for images, targets in train_loader:

            #move data to device for efficiency, and to ensure model and tensors are on the same device
            #non_blocking=True allows data transfer to be asynchronous, so CPU can continue loading data while GPU is training the model
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            #clear gradients from previous iteration, otherwise they will accumulate and cause incorrect updates to the model parameters
            optimizer.zero_grad(set_to_none=True)

            #forward pass through the model, which returns predictions for the input images
            predictions = model(images)

            #calculate loss between predictions and targets
            loss = detection_loss(predictions, targets)

            #backward pass through model to calculate gradients of the loss with respect to the model parameters
            #if a weight causes a large loss, its gradient will be large
            loss.backward()

            #loss.backward() calculates how much weights should be changed, and optimizer.step() updates the weights
            optimizer.step()

            #accumulate loss for the epoch, to calculate average loss at the end of the epoch
            running_loss += loss.item()

            #increment batch counter
            current_batch += 1

        #calculate average loss for the epoch, to monitor training progress
        average_loss = (running_loss / len(train_loader))

        #print training and validation metrics for the epoch, to monitor training progress
        print(
            #format is "metric_name = metric_value", with 4 decimal places for loss, precision, and recall
            f"training_loss = {average_loss:.4f}, "
        )

        #use moduli function to validate after a certain number of epochs
        if (epoch + 1) % 1 == 0:
            print("Validating...")

            #validate, and return validation loss
            val_loss = Validation(model, val_loader=val_loader, device=DEVICE)

            #check for best valuation loss and save model if its good
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                #saves the model state dictionary and training parameters after a succesful validation, 
                #so training can be resumed later if interrupted 
                checkpoint = {      
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),   
                            }

                #we use .pt instead of .pth to prevent confusion with pythons path files
                torch.save(checkpoint, root_directory / checkpoint_file)

                print(
                    f"New best teacher saved! "
                )


#call execution of main function if this script is run directly, rather than imported as a module
if __name__ == "__main__":
    main()