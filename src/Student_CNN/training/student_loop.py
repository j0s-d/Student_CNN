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
from torchvision.ops import box_iou, nms

#---------- Load checkpoint if it exists ----------

def LoadCheckpoint(student_model, teacher_model, optimizer, scheduler, filename, student, teacher):

    #checks if checkpoint file exists
    if os.path.isfile(filename / student):

        print(f"Loading student checkpoint")

        #loads each component in the correct order- very important
        checkpoint = torch.load(filename / student)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']

        #print start epoch and checkpoint filename, to confirm that the checkpoint was loaded correctly
        print(f"Loaded student checkpoint '{filename} / {student}' (epoch {start_epoch})")

        #if using knowledge distillation, load teacher model
        if teacher_model != None:
            if os.path.isfile(filename / teacher):
                print(f"Loading teacher checkpoint")

                checkpoint = torch.load(filename / teacher)
                teacher_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded teacher checkpoint '{filename} / {teacher}'")

        return start_epoch, best_f1

    else:
        print(f"No student checkpoint found at '{filename}'")
        return 0, float(0)

#------- Decode targets -------

def DecodeTargets(targets, grid_size):

    device = targets.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij"
    )

    grid_x = grid_x.reshape(1, -1)
    grid_y = grid_y.reshape(1, -1)

    obj = targets[..., 0]

    rel_x = targets[..., 1]
    rel_y = targets[..., 2]

    width = targets[..., 3]
    height = targets[..., 4]

    cx = (grid_x + rel_x) / grid_size
    cy = (grid_y + rel_y) / grid_size

    return torch.stack([
        obj,
        cx,
        cy,
        width,
        height
    ], dim=-1)

#---------- Decode different detection heads separately --------

def DecodeHead(pred, grid_size):
    B, N, C = pred.shape

    expected_cells = grid_size * grid_size

    assert N == expected_cells, (
        f"Expected {expected_cells} cells for "
        f"{grid_size}x{grid_size} head, got {N}"
    )

    assert C == 5, (
        f"Expected 5 prediction values, got {C}"
    )

    #reshape flattened head back to grid
    pred = pred.reshape(B, grid_size, grid_size, 5)

    #objectness
    confidence = torch.sigmoid(pred[..., 0])

    #relative position inside cell
    tx = torch.sigmoid(pred[..., 1])
    ty = torch.sigmoid(pred[..., 2])

    #width / height relative to image
    width = torch.sigmoid(pred[..., 3])
    height = torch.sigmoid(pred[..., 4])

    #create grid coordinates
    device = pred.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij"
    )

    grid_x = grid_x.float()
    grid_y = grid_y.float()

    #convert cell-relative coordinates
    #into image-relative coordinates
    cx = (grid_x + tx) / grid_size
    cy = (grid_y + ty) / grid_size

    #stack boxes into a tensor
    decoded = torch.stack(
        [
            confidence,
            cx,
            cy,
            width,
            height
        ],
        dim=-1
    )

    #puts predictions in form [B, N, C]
    return decoded.reshape(B, -1, 5)

#---------- Decode precitions for validation metrics ----------

def IoUAnalysis(predictions, 
                targets, 
                confidence_threshold = 0.9, 
                nms_threshold = 0.1, 
                iou_threshold = 0.3):
    
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
    batch, _,  _ = predictions.shape

    #initialise stats to calculate precision/recall
    tp = 0
    fp = 0
    fn = 0

    for b in range(batch):

        #predictions/targets for individual images
        pred = predictions[b]
        target = targets[b]

        #only use predictions above confidence threshold by applying mask 
        pred_mask = pred[:, 0] >= confidence_threshold
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
        pred_cx = pred[:, 1]
        pred_cy = pred[:, 2]
        pred_width = pred[:, 3]
        pred_height = pred[:, 4]

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

        #apply nms
        scores = pred[:, 0]
        keep = nms(pred_boxes, scores, nms_threshold)

        pred_boxes = pred_boxes[keep] 
        pred = pred[keep] 

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
    with torch.inference_mode():

        for images, small_targets, medium_targets, large_targets in val_loader:

            
            #move validation data to device used for training
            images = images.to(
                device,
                non_blocking=True
            )

            small_targets = small_targets.to(
                device,
                non_blocking=True
            )

            medium_targets = medium_targets.to(
                device,
                non_blocking=True
            )

            large_targets = large_targets.to(
                device,
                non_blocking=True
            )

            #concatrate targets 
            targets = torch.cat([small_targets, medium_targets, large_targets], dim=1)

            #forward pass only through the model
            small_faces, medium_faces, large_faces = model(images)

            raw_predictions = torch.cat([small_faces, medium_faces, large_faces], dim=1)
            
            #calculate validation loss
            loss, _ = detection_loss(
                raw_predictions,
                targets
            )

            #tally validation loss
            validation_loss += loss.item()

            small = DecodeHead(small_faces, 224 // 8)
            medium = DecodeHead(medium_faces, 224 // 16)
            large = DecodeHead(large_faces, 224 // 32)

            small_target = DecodeTargets(small_targets, 224 // 8)
            medium_target = DecodeTargets(medium_targets, 224 // 16)
            large_target = DecodeTargets(large_targets, 224 // 32)

            decoded_predictions = torch.cat([small, medium, large], dim=1)
            decoded_targets = torch.cat([small_target, medium_target, large_target], dim=1)

            #calculate statistics for precision/recall and tally
            tp, fp, fn = IoUAnalysis(decoded_predictions, decoded_targets)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

    #average validation loss across batches
    average_val_loss = validation_loss / len(val_loader)

    #calculate precision/recall and avoid division by 0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0

    #calculate f1
    f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0.0
    )

    #print statistics, keeping to 4 significant figures
    print(
        f"Validation_loss = {average_val_loss:.4f}, "
        f"f1 = {f1:.4f}, "
        f"Validation_precision = {precision:.4f}, "
        f"Validation_recall = {recall:.4f}"
    )

    return average_val_loss, precision, recall, f1

#put all code in a main function, so that it can be called when the script is run directly 
#but not when it is imported as a module
def main():

    #find root directory for later
    root_directory = Path(__file__).parent.parent

    #EPOCHS is the number of times the model will see the entire dataset during training
    EPOCHS = 100

    #knowledge distillation bool
    know_diss = True

    #set best model parameters
    best_f1 = float(0)

    #initialses lists to store training data for plotting later
    if know_diss:
        csv_directory = root_directory / "training" / "history_kd"
    else:
        csv_directory = root_directory / "training" / "history"
    
    history_train_loss = []
    history_train_loss_kd = []
    history_val_loss = []
    history_f1 = []
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
        batch_size=32,
        shuffle=True,
        num_workers=4, #load data in parallel using 4 worker threads, recommended at 4 * no. GPU's
        
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
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        #keeps workers running over each epoch
        persistent_workers=True
    )

    #puts model onto device
    student_model = StudentFaceDetector().to(DEVICE)

    if know_diss:
        teacher_model = TeacherFaceDetector().to(DEVICE)
    else:
        teacher_model = None


    #creates an AdamW optimiser for the model parameters
    #weight decay is a regularisation technique that reduces overfitting by penalising large weights
    #this prevents the model from memorising the training data, and encourages it to learn generalisable features
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-3)

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

    start_epoch, best_f1 = LoadCheckpoint(
        student_model, 
        teacher_model, 
        optimizer, 
        scheduler,
        filename=root_directory, 
        student=student_checkpoint_file, 
        teacher= teacher_checkpoint_file
        )


    #---------- Main loop ----------

    for epoch in range(start_epoch, EPOCHS):

        #----------- Training loop ----------

        #print epoch number, to monitor training progress
        print(f"Epoch {epoch + 1}/{EPOCHS} ... Training ...")

        #puts model into training mode, which enables dropout and batch normalisation layers to behave differently during training and evaluation
        #for example, nn.BatchNorm2d uses the mean and variance of the current batch during training, but uses the running mean and variance during evaluation
        student_model.train()

        #set running loss and accuracy to 0 for each epoch, to calculate average at the end of the epoch
        running_loss = 0.0
        running_loss_kd = 0.0

        #initialize batch counter, to monitor training progress
        current_batch = 0

        for images, small_targets, medium_targets, large_targets in train_loader:

            #move data to device for efficiency, and to ensure model and tensors are on the same device
            #non_blocking=True allows data transfer to be asynchronous, so CPU can continue loading data while GPU is training the model
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            small_targets = small_targets.to(
                DEVICE,
                non_blocking=True
            )

            medium_targets = medium_targets.to(
                DEVICE,
                non_blocking=True
            )

            large_targets = large_targets.to(
                DEVICE,
                non_blocking=True
            )

            #concatrate targets 
            targets = torch.cat([small_targets, medium_targets, large_targets], dim=1)

            #clear gradients from previous iteration, otherwise they will accumulate and cause incorrect updates to the model parameters
            optimizer.zero_grad(set_to_none=True)

            #forward pass through the model, which returns predictions for the input images
            small_faces, medium_faces, large_faces = student_model(images)

            teacher_predictions=None

            #forward pass over teacher, to use its predictions in knowledge distillation
            if know_diss:
                with torch.no_grad():
                    t_small_faces, t_medium_faces, t_large_faces = teacher_model(images)
                    teacher_predictions = torch.cat([t_small_faces, t_medium_faces, t_large_faces], dim=1)
                

            raw_predictions = torch.cat([small_faces, medium_faces, large_faces], dim=1)

            #calculate loss between predictions and targets
            loss, know_diss_loss = detection_loss(raw_predictions, targets, teacher_prediction=teacher_predictions)

            #backward pass through model to calculate gradients of the loss with respect to the model parameters
            #if a weight causes a large loss, its gradient will be large
            loss.backward()

            #loss.backward() calculates how much weights should be changed, and optimizer.step() updates the weights
            optimizer.step()

            #accumulate loss for the epoch, to calculate average loss at the end of the epoch
            running_loss += loss.item()
            running_loss_kd += know_diss_loss.item()

            #increment batch counter
            current_batch += 1

        #calculate average loss for the epoch, to monitor training progress
        average_loss = (running_loss / len(train_loader))
        average_loss_kd = (running_loss_kd / len(train_loader))

        #append to history list
        history_train_loss.append(average_loss)
        history_train_loss_kd.append(average_loss_kd)

        #print training and validation metrics for the epoch, to monitor training progress
        print(
            #format is "metric_name = metric_value", with 4 decimal places for loss, precision, and recall
            f"training_loss = {average_loss:.4f}, "
            f"training_loss_kd = {average_loss_kd:.4f}, "
        )

        #update learning rate after each epoch, to help the model converge to a minimum
        scheduler.step()

        #use moduli function to validate after a certain number of epochs
        if (epoch + 1) % 1 == 0:
            print("Validating...")

            #validate, and return validation loss
            val_loss, precision, recall, f1 = Validation(student_model, val_loader=val_loader, device=DEVICE)

            #append values to history lists
            history_val_loss.append(val_loss)
            history_precision.append(precision)
            history_recall.append(recall)
            history_f1.append(f1)

            #check for best valuation loss and save model if its good
            if f1 > best_f1:

                best_f1 = f1

                #saves the model state dictionary and training parameters after a succesful validation, 
                #so training can be resumed later if interrupted 
                checkpoint = {      
                            'epoch': epoch + 1,
                            'model_state_dict': student_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(), 
                            'best_f1': best_f1 
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
    plt.savefig(csv_directory / "train/train_loss.jpg")
    plt.show()

    plt.figure(2)
    plt.plot(history_train_loss_kd, 'm-')
    plt.title("Training Loss KD")
    plt.savefig(csv_directory / "train/train_loss_kd.jpg")
    plt.show()

    plt.figure(3)
    plt.plot(history_val_loss, 'g-')
    plt.title("Validation Loss")
    plt.savefig(csv_directory / "val/val_loss.jpg")
    plt.show()

    plt.figure(4)
    plt.plot(history_f1, 'p-')
    plt.title("F1")
    plt.savefig(csv_directory / "val/f1.jpg")
    plt.show()

    plt.figure(5)
    plt.plot(history_precision, 'r-')
    plt.title("Precision")
    plt.savefig(csv_directory / "val/precision.jpg")
    plt.show()

    plt.figure(6)
    plt.plot(history_recall, 'b-')
    plt.title("Recall")
    plt.savefig(csv_directory / "val/recall.jpg")
    plt.show()
    

    #save history lists as csv files
    np.savetxt(csv_directory / "train/train_loss.csv", history_train_loss, delimiter=",")
    np.savetxt(csv_directory / "train/train_loss_kd.csv", history_train_loss_kd, delimiter=",")
    np.savetxt(csv_directory / "val/val_loss.csv", history_val_loss, delimiter=",")
    np.savetxt(csv_directory / "val/f1.csv", history_f1, delimiter=",")
    np.savetxt(csv_directory / "val/precision.csv", history_precision, delimiter=",")
    np.savetxt(csv_directory / "val/recall.csv", history_recall, delimiter=",")

#call execution of main function if this script is run directly, rather than imported as a module
if __name__ == "__main__":
    main()