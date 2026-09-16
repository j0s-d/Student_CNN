#import modules
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from Student_CNN.student_model.student import StudentFaceDetector
from Student_CNN.teacher_model.teacher import TeacherFaceDetector
from . dataset import FaceAsTensorDataset
from . loss_function import detection_loss
from torchvision.ops import box_iou

#---------- Load checkpoint if it exists ----------

def LoadCheckpoint(student_model, teacher_model, optimizer, filename, student, teacher):

    #checks if checkpoint file exists
    if os.path.isfile(filename / student):

        print(f"Loading student checkpoint")

        #loads each component in the correct order- very important
        checkpoint = torch.load(filename / student)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        #print start epoch and checkpoint filename, to confirm that the checkpoint was loaded correctly
        print(f"Loaded student checkpoint '{filename} / {student}' (epoch {start_epoch})")

        if os.path.isfile(filename / teacher):
            print(f"Loading teacher checkpoint")

            checkpoint = torch.load(filename / teacher)
            teacher_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded teacher checkpoint '{filename} / {teacher}'")

        return start_epoch

    else:
        print(f"No student checkpoint found at '{filename}'")
        return 0

#---------- Decode precitions for validation metrics ----------

def IoUAnalysis(predictions, targets, confidence_threshold = 0.8, iou_threshold = 0.3):
    
    #sanity checks
    if predictions.dim() != 3:
        raise ValueError(
            f"Expected predictions [B,N,5], "
            f"got {predictions.shape}"
        )

    if targets.dim() != 3:
        raise ValueError(
            f"Expected targets [B,N,5], "
            f"got {targets.shape}"
        )
    
    #tensor shape from predictions
    batch, width, height, _ = predictions.shape

    #initialise stats to calculate precision/recall
    tp = 0
    fp = 0
    fn = 0

    for b in range(batch):

        #predictions/targets for individual images
        pred = predictions[b]
        target = targets[b]

        #only use predictions above confidence threshold by applying mask 
        pred_mask = torch.sigmoid(pred[:, 0]) >= confidence_threshold
        pred = pred[pred_mask]

        #gather positive ground truth cells
        target_mask = target[:, 0] == 1
        target = target[target_mask]

        #no predictions/targets, tp, fp, fn all equal 0
        if len(pred) == 0 and len(target) == 0:
            continue

        #no predictions, fn equal no. targets
        if len(pred) == 0:
            fn += len(target)
            continue

        #no targets, fp = no. predictions
        if len(target) == 0:
            fp += len(pred)
            continue

        #convert model probabilities to coordinates
        pred_cx = torch.sigmoid(pred[:, 1])
        pred_cy = torch.sigmoid(pred[:, 2])
        pred_width = torch.sigmoid(pred[:, 3])
        pred_height = torch.sigmoid(pred[:, 4])

        #create box with absolute coordinates
        pred_boxes = torch.stack([
            pred_cx - pred_width / 2,
            pred_cy - pred_height / 2,
            pred_cx + pred_width / 2,
            pred_cy + pred_height / 2,
        ], dim=1)

        #extract coordinates from target
        target_cx = target[:, 1]
        target_cy = target[:, 2]
        target_width = target[:, 3]
        target_height = target[:, 4]

        #create box with absolute target coordinates
        target_boxes = torch.stack([
            target_cx - target_width / 2,
            target_cy - target_height / 2,
            target_cx + target_width / 2,
            target_cy + target_height / 2,
        ], dim=1)

        ious = box_iou(pred_boxes, target_boxes)

        matched_predictions = set()
        matched_targets = set()

        match_sort = torch.argsort(ious.flatten(), descending=True)

        for flat in match_sort:
            p = (flat // ious.shape[1]).item()
            t = (flat % ious.shape[1]).item()

            #only count ious that overlap greater than threshold
            if ious[p, t] < iou_threshold:
                break

            #prevents predictions and targets being matched more than once
            if p in matched_predictions or t in matched_targets:
                continue

            matched_predictions.add(p)
            matched_targets.add(t)

        #summary of statistics for the batch
        batch_tp = len(matched_predictions)
        batch_fp = len(pred) - batch_tp
        batch_fn = len(target) - batch_tp

        #tally batch to total statistics
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn

    return tp, fp, fn

#---------- Validation loop ----------

def Validation(model, val_loader, device="cpu"):

    #set model into evaluation mode, which affects the behaviour of the batchnorm2d block
    model.eval()

    #initialise variables
    validation_loss = 0.0
    true_positives = 0
    false_positives = 0
    false_negatives = 0


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

            #calculate statistics for precision/recall and tally
            tp, fp, fn = IoUAnalysis(predictions, targets)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

    #average validation loss across batches
    average_val_loss = validation_loss / len(val_loader)

    #calculate precision/recall and avoid division by 0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0


    #print statistics, keeping to 4 significant figures
    print(
        f"Validation_loss = {average_val_loss:.4f}, "
        f"Validation_precision = {precision:.4f}, "
        f"Validation_recall = {recall:.4f}"
    )

    return average_val_loss, precision, recall

#put all code in a main function, so that it can be called when the script is run directly 
#but not when it is imported as a module
def main():

    #find root directory for later
    root_directory = Path(__file__).parent.parent

    #EPOCHS is the number of times the model will see the entire dataset during training
    EPOCHS = 30

    #initialses lists to store training data for plotting later
    csv_directory = root_directory / "training" / "history"
    history_train_loss = []
    history_val_loss = []
    history_precision = []
    history_recall = []

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
        image_dir=root_directory.parent.parent / "data" / "WIDER_JSON" / "train" / "images",
        label_dir=root_directory.parent.parent / "data" / "WIDER_JSON" / "train" / "labels",
        input_size=224,
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
        image_dir=root_directory.parent.parent / "data" / "WIDER_JSON" / "val" / "images",
        label_dir=root_directory.parent.parent / "data" / "WIDER_JSON" / "val" / "labels",
        input_size=224,
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
    student_model = StudentFaceDetector().to(DEVICE)
    teacher_model = TeacherFaceDetector().to(DEVICE)


    #creates an AdamW optimiser for the model parameters
    #weight decay is a regularisation technique that reduces overfitting by penalising large weights
    #this prevents the model from memorising the training data, and encourages it to learn generalisable features
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.001, weight_decay=1e-5)

    #creates a cosine annealing learning rate scheduler 
    #reduces the learning rate over time to help the model converge to a minimum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    #initialize start_epoch to 0, will be updated if checkpoint is loaded
    start_epoch = 0

    #set patience counter to 0
    patience_counter = 0
    patience = 5

    #load checkpoint files
    student_checkpoint_file = "student_checkpoint.pt"
    teacher_checkpoint_file = "teacher_checkpoint.pt"

    start_epoch = LoadCheckpoint(
        student_model, 
        teacher_model, 
        optimizer, 
        filename=root_directory, 
        student=student_checkpoint_file, 
        teacher= teacher_checkpoint_file
        )


    #set best model parameters
    best_val_loss = float("inf")


    #---------- Main loop ----------

    for epoch in range(start_epoch, EPOCHS):

        #----------- Training loop ----------

        #print epoch number, to monitor training progress
        print(f"Epoch {epoch + 1}/{EPOCHS} ... Training ...")

        #puts model into training mode, which enables dropout and batch normalisation layers to behave differently during training and evaluation
        #for example, nn.BatchNorm2d uses the mean and variance of the current batch during training, but uses the running mean and variance during evaluation
        student_model.train()

        
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
            predictions = student_model(images)

            #forward pass over teacher, to use its predictions in knowledge distillation
            teacher_predicitions = teacher_model(images)

            #calculate loss between predictions and targets
            loss = detection_loss(predictions, targets, teacher_predicion=teacher_predicitions)

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

        #append to history list
        history_train_loss.append(average_loss)

        #print training and validation metrics for the epoch, to monitor training progress
        print(
            #format is "metric_name = metric_value", with 4 decimal places for loss, precision, and recall
            f"training_loss = {average_loss:.4f}, "
        )

        #use moduli function to validate after a certain number of epochs
        if (epoch + 1) % 1 == 0:
            print("Validating...")

            #validate, and return validation loss
            val_loss, precision, recall = Validation(student_model, val_loader=val_loader, device=DEVICE)

            #append values to history lists
            history_val_loss.append(val_loss)
            history_precision.append(precision)
            history_recall.append(recall)

            #check for best valuation loss and save model if its good
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                #saves the model state dictionary and training parameters after a succesful validation, 
                #so training can be resumed later if interrupted 
                checkpoint = {      
                            'epoch': epoch + 1,
                            'model_state_dict': student_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),   
                            }

                #we use .pt instead of .pth to prevent confusion with pythons path files
                torch.save(checkpoint, root_directory / student_checkpoint_file)

                print(
                    f"New best student saved! "
                )

                #reset patience counter
                patience_counter = 0

            else:

                #increment patience counter
                patience_counter += 1

        #loop stops early if model does not improve
        if patience_counter >= patience:
            print("Early stopping")
            break


    #--------- Plot and save data --------

    plt.figure(1)
    plt.plot(history_train_loss, 'o-')
    plt.title("Training Loss")
    plt.savefig(csv_directory / "training/train_loss.jpg")
    plt.show()

    plt.figure(2)
    plt.plot(history_val_loss, 'g-')
    plt.title("Validation Loss")
    plt.savefig(csv_directory / "val/val_loss.jpg")
    plt.show()

    plt.figure(3)
    plt.plot(history_precision, 'r-')
    plt.title("Precision")
    plt.savefig(csv_directory / "val/precision.jpg")
    plt.show()

    plt.figure(4)
    plt.plot(history_recall, 'b-')
    plt.title("Recall")
    plt.savefig(csv_directory / "val/recall.jpg")
    plt.show()
    

    #save history lists as csv files
    np.savetxt(csv_directory / "training/train_loss.csv", history_train_loss, delimiter=",")
    np.savetxt(csv_directory / "val/val_loss.csv", history_val_loss, delimiter=",")
    np.savetxt(csv_directory / "val/precision.csv", history_precision, delimiter=",")
    np.savetxt(csv_directory / "val/recall.csv", history_recall, delimiter=",")

#call execution of main function if this script is run directly, rather than imported as a module
if __name__ == "__main__":
    main()