import torch
from data_preprocess import seed_torch, get_datasets,get_dataloader, train, test
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

def train_teacher_model(Teacher_model,data_name,Teacher_training_epochs,dataloader,path,device):
    criterion = nn.CrossEntropyLoss().to(device)
    if data_name == 'cifar100':
        optimizer = optim.Adam(Teacher_model.parameters(), lr=1e-3, weight_decay = 5e-4)
    else:
        optimizer = optim.Adam(Teacher_model.parameters(), lr=1e-3)
    epoch_loss = 0
    epoch_acc = 0
    for epoch in range(1, Teacher_training_epochs + 1):
        epoch_loss, epoch_acc = train(Teacher_model, dataloader, device, optimizer, criterion)
        print('[Train #{}] Train Loss: {:.4f} Train Acc: {:.4f}% '.format(epoch, epoch_loss, epoch_acc))
    torch.save({
                    'model_state_dict': Teacher_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
    return Teacher_model


def GMUUD_Unlearning(GMUUD_params,Teacher1_model,Teacher2_model,Student_model,remaining_dataloader,unlearning_dataloader,optimizer,device):

    # Distillation losses
    hard_loss = nn.CrossEntropyLoss()
    soft_loss = nn.KLDivLoss(reduction="batchmean")
    KL_loss = nn.KLDivLoss(reduction="batchmean")
    mse_loss =nn.MSELoss()
    # Knowledge Distillation
    # epoch_loss = 0
    # epoch_acc = 0
    for epoch in range(1, GMUUD_params.KD_epochs + 1):

        Student_model.train()
        # On unlearning data
        for i, (inputs, labels) in enumerate(tqdm(unlearning_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            Student_outputs = Student_model(inputs)
            Teacher1_model.eval()
            Teacher2_model.eval()
            with torch.no_grad():
                Teacher_outputs = Teacher1_model(inputs)
                Teacher2_outputs = Teacher2_model(inputs)

            loss = -GMUUD_params.gama*(KL_loss(F.log_softmax(Teacher_outputs, dim=1), F.softmax(Student_outputs, dim=1)) + KL_loss(F.log_softmax(Teacher2_outputs, dim=1), F.softmax(Student_outputs, dim=1)))
            # total loss
            preds = Student_outputs.argmax(dim=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        # On remaining data
        # train_loss = 0.
        # train_corrects = 0
        # KD_loss = 0.
        # G_loss = 0.

        for i, (inputs, labels) in enumerate(tqdm(remaining_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            Student_outputs = Student_model(inputs)
            Teacher1_model.eval()
            Teacher2_model.eval()
            with torch.no_grad():
                Teacher_outputs = Teacher1_model(inputs)
                Teacher2_outputs = Teacher2_model(inputs)


            loss_hard = hard_loss(Student_outputs, labels)
            ditillation_loss = soft_loss(F.log_softmax(Student_outputs / GMUUD_params.T, dim=1), F.softmax(Teacher_outputs / GMUUD_params.T, dim=1))
            loss_KD = GMUUD_params.alpha * loss_hard + ditillation_loss * GMUUD_params.beta
            loss_g = KL_loss(F.log_softmax(Student_outputs, dim=1), F.softmax(Teacher2_outputs, dim=1))

            loss_reg = mse_loss(KL_loss(F.log_softmax(Teacher2_outputs, dim=1), F.softmax(Student_outputs, dim=1)),KL_loss(F.log_softmax(Teacher2_outputs, dim=1), F.softmax(Teacher_outputs, dim=1)))
            


            # total loss
            loss = loss_KD - GMUUD_params.gama*loss_g + GMUUD_params.lamda*loss_reg

            preds = Student_outputs.argmax(dim=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_loss += loss.item() * inputs.size(0)
            # train_corrects += torch.sum(preds == labels.data)
            # KD_loss += loss_KD.item() * inputs.size(0)
            # G_loss += loss_g.item() * inputs.size(0)

        # epoch_loss = train_loss / len(remaining_dataloader.dataset)
        # epoch_acc = train_corrects / len(remaining_dataloader.dataset) * 100.
        # epoch_KD_loss = KD_loss / len(remaining_dataloader.dataset)
        # epoch_G_loss = G_loss / len(remaining_dataloader.dataset)
        # print('[KD #{}] Student Loss: {:.4f} Student Acc: {:.4f}% KD_Loss: {:.4f}  G_Loss: {:.4f}'.format(epoch,
        #                                                                                                         epoch_loss,
        #                                                                                                         epoch_acc,
        #                                                                                                         epoch_KD_loss,epoch_G_loss))


    if GMUUD_params.finetune_epochs:
        epoch_loss = 0
        epoch_acc = 0
        criterion = nn.CrossEntropyLoss().to(device)
        for epoch in range(1, GMUUD_params.finetune_epochs + 1):
            epoch_loss, epoch_acc = train(Student_model, remaining_dataloader, device, optimizer, criterion)
            print('[Finetune #{}] Train Loss: {:.4f} Train Acc: {:.4f}% '.format(epoch, epoch_loss, epoch_acc))
    return Student_model

def Acc_eval(Student_model, unlearning_dataloader,remaining_dataloader, test_dataloader,device):
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss, test_acc = test(Student_model, unlearning_dataloader, criterion, device)
    print('[Student model] Unlearning Loss: {:.4f} Unlearning Acc: {:.4f}% '.format(test_loss, test_acc))
    test_loss, test_acc = test(Student_model, remaining_dataloader, criterion, device)
    print('[Student model] Remaining Loss: {:.4f} Remaining Acc: {:.4f}% '.format(test_loss, test_acc))
    test_loss, test_acc = test(Student_model, test_dataloader, criterion, device)
    print('[Student model] Test Loss: {:.4f} Test Acc: {:.4f}% '.format(test_loss, test_acc))