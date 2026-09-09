#import modules
import torch
import torch.nn.functional as func

#--------- Simple loss function for training the face detection network ----------

def detection_loss(prediction, 
                   target, 
                   teacher_predicion=None, 
                   lambda_box=2.0, 
                   lambda_noobj=0.5, 
                   lambda_wh = 1.5, 
                   lambda_kd=1, 
                   temperature=5 
                   ):

    #seperate the objectness score and bounding box coordinates from the prediction and target tensors
    #this allows two types of loss to be calculated, one for the objectness score and one for the bounding box coordinates
    pred_obj = prediction[ ... , 0]
    pred_box = prediction[ ... , 1:]
    target_obj = target[ ... , 0]
    target_box = target[ ... , 1:]

    if teacher_predicion != None:
        teacher_predicion = teacher_predicion.detach() #use this so loss doesn't backpropogate through teacher model
        teach_obj = teacher_predicion[ ... , 0]
        teach_box = teacher_predicion[ ... , 1:]




    #efficient why of calculating whether there is a face in the cell or not, by checking if the objectness score is 1 or 0
    #positive variable returns True of target_obj is 1 eg.
    positive = target_obj == 1
    negative = target_obj == 0

    #logit function ---> logit(p) = ln(p/(1−p))
    #logit function is the inverse of the sigmoid function (maps any real value to a value between 0 and 1)
    #logit function is used to convert the objectness score from a probability to a logit, which is more suitable for calculating the loss
    #binary cross entropy loss is used to calculate the loss for the objectness score, as there is either a face or not in the cell
    pos_loss = func.binary_cross_entropy_with_logits(
        pred_obj[positive],
        target_obj[positive],
        reduction="sum"
    )

    neg_loss = func.binary_cross_entropy_with_logits(
        pred_obj[negative],
        target_obj[negative],
        reduction="sum"
    )

    #scales the positive and negative losses by the number of positive and negative cells respectively, to get the average loss per cell
    #clamp prevents division by zero
    num_positive = positive.sum().clamp(min=1)
    num_negative = negative.sum().clamp(min=1)

    pos_loss = pos_loss / num_positive
    neg_loss = neg_loss / num_negative

    #total loss for the objectness score is the sum of the positive and negative losses, with the negative loss being weighted by lambda_noobj
    #there are likely to be way more negative cells than positive cells
    #this weighting prevents the model from being biased towards predicting no face in the cell (if negative loss dominates)
    obj_loss = (
        pos_loss + lambda_noobj * neg_loss
    )

    #calculate the loss for the bounding box coordinates, only for the cells that contain a face (positive cells)
    #computationally efficient
    if positive.any():

        #smooth L1 loss is used to calculate the loss for the bounding box coordinates
        #behaves like absolute error for large errors and like squared error for small errors, making it more robust to outliers
        #smooth L1 loss is also known as Huber loss, and is a combination of L1 and L2 loss
        #sigmoid function is used to convert the predicted bounding box coordinates from logits to probabilities 
        #these are then compared to the target bounding box coordinates
        pred_xy = torch.sigmoid(pred_box[..., 0:2])
        pred_wh = torch.sigmoid(pred_box[..., 2:4])

        xy_loss = func.smooth_l1_loss(
            pred_xy[positive],
            target_box[positive, 0:2],
            reduction="sum"
        ) / num_positive

        wh_loss = func.smooth_l1_loss(
            pred_wh[positive],
            target_box[positive, 2:4],
            reduction="sum"
        ) / num_positive

        box_loss = xy_loss + lambda_wh * wh_loss

    else:
        box_loss = torch.tensor(0.0, device=prediction.device)


    #---------- Knowledge Distillation --------

    knowdiss_loss = torch.tensor(0.0, device=prediction.device)

    if teacher_predicion != None:

        #calculate soft targets, weighted by a temperature parameter
        student_soft_obj = torch.sigmoid(pred_obj / temperature)

        #calculate soft targets, weighted by a temperature parameter
        teacher_soft_obj = torch.sigmoid(teach_obj / temperature)

        obj_knowdiss_loss = func.binary_cross_entropy_with_logits(
            student_soft_obj,
            teacher_soft_obj
        )

        if positive.any():

            #retrieve teacher bounding box coordinates
            teach_xy = torch.sigmoid(teach_box[..., 0:2])
            teach_wh = torch.sigmoid(teach_box[..., 2:4])

            #caluclate box loss as previously
            xy_knowdiss_loss = func.smooth_l1_loss(
                pred_xy[positive],
                teach_xy[positive],
                reduction="sum"
            ) / num_positive

            wh_knowdiss_loss = func.smooth_l1_loss(
                pred_wh[positive],
                teach_wh[positive],
                reduction="sum"
            ) / num_positive

            #calculate box loss, weighting wh loss by lambda_wh
            box_knowdiss_loss = xy_knowdiss_loss + lambda_wh * wh_knowdiss_loss

        else:

            #set box loss to zero if no boxes predicted
            box_knowdiss_loss = torch.tensor(0.0, device= prediction.device)

        #calculate final knowledge distillation loss, box loss weighted by lambda_box
        knowdiss_loss = obj_knowdiss_loss + lambda_box * box_knowdiss_loss

    #bounding box loss is weighted by lambda_box, as this is the most important part of the loss function, as it is the most difficult to learn
    total = obj_loss + lambda_box * box_loss + lambda_kd * knowdiss_loss
        

    return total 