
import torch
from data_preprocess import seed_torch, get_datasets,get_dataloader, train, test, dataset_prune
import torch.nn as nn
import torch.optim as optim
from model_initiation import get_model
from GMUUD_base import GMUUD_Unlearning,train_teacher_model,Acc_eval

class Arguments():
    def __init__(self):
        #KD unlearning Settings
        self.data_name = 'cifar10' # cifar100, mnist
        self.num_classes = 10
        self.seed_num = 500
        
        self.data_dir = './Dataset/{}_dataset'.format(self.data_name)
        # Whether to use data pruning
        self.use_data_prune = False
        self.num_perclass = 100
        self.data_prune_path = './Dataset/Prune/{}dataset_{}perclass_IS.pt'.format(self.data_name, self.num_perclass)
        
        
        self.numclass = 10
        self.batch_size=256
        # teacher1 is the origin model, teacher2 is the gudiance model
        self.teacher1_model_name = 'ResNet18'
        self.teacher1_training_epochs = 30

        self.teacher2_model_name = 'ResNet8'
        self.teacher2_training_epochs = 100

        self.Teacher1_model_train = False
        self.Teacher2_model_train = False
        self.teacher1_path = "./Model/Teacher1/Teacher1_model_{}_{}_{}_{}.pt".format(self.teacher1_model_name,self.teacher1_training_epochs, self.seed_num, self.data_name)
        self.teacher2_path = "./Model/Teacher2/Teacher2_model_{}_{}_{}_{}.pt".format(self.teacher2_model_name,self.teacher2_training_epochs, self.seed_num, self.data_name)


        self.Student_model_name = 'ResNet18'
        # Distillation Settings
        self.KD_epochs = 10
        self.T = 9
        self.alpha = 1.5
        self.beta = 0.9
        self.gama = 0.0001
        self.lamda = 0.0001
        self.finetune_epochs = 3

        
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True

       



def GMUUD_Unlearning_main():
    GMUUD_params = Arguments()
    device = torch.device("cuda" if GMUUD_params.use_gpu*GMUUD_params.cuda_state else "cpu")
    device_cpu = torch.device("cpu")
    seed_torch(GMUUD_params.seed_num)
    # Loading data
    train_datasets, test_datasets = get_datasets(GMUUD_params.data_name, GMUUD_params.data_dir)
    remaining_dataloader, test_dataloader, unlearning_dataloader, total_dataloader = get_dataloader(train_datasets, test_datasets,GMUUD_params.batch_size)
    
    # Training/Loading the original model as teacher
    Teacher1_model = get_model(GMUUD_params.teacher1_model_name, GMUUD_params.numclass).to(device)
    if GMUUD_params.Teacher1_model_train:
        Teacher1_model = train_teacher_model(Teacher1_model,GMUUD_params.data_name,GMUUD_params.Teacher1_training_epochs,total_dataloader,GMUUD_params.teacher1_path,device)

    else:
        checkpoint = torch.load(GMUUD_params.teacher1_path)
        Teacher1_model.load_state_dict(checkpoint['model_state_dict'])


    # Training/Loading auxiliary models
    
    Teacher2_model = get_model(GMUUD_params.teacher2_model_name, GMUUD_params.numclass).to(device)
    if GMUUD_params.Teacher2_model_train:
        Teacher2_model = train_teacher_model(Teacher2_model,GMUUD_params.data_name,GMUUD_params.Teacher2_training_epochs,unlearning_dataloader,GMUUD_params.teacher2_path,device)
    else:
        checkpoint = torch.load(GMUUD_params.teacher2_path)
        Teacher2_model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize Student model with same structure as Teacher1
    Student_model_name = GMUUD_params.teacher1_model_name
    Student_model = get_model(Student_model_name, GMUUD_params.numclass).to(device)
    # optimizer = optim.Adam(Student_model.parameters(), lr=1e-3)
    if GMUUD_params.data_name == 'cifar100':
        optimizer = optim.Adam(Student_model.parameters(), lr=1e-3, weight_decay = 5e-4)
    else:
        optimizer = optim.Adam(Student_model.parameters(), lr=1e-3)
    
    if GMUUD_params.use_data_prune:
        dataset_prune(train_datasets,GMUUD_params.num_classes,Teacher1_model,GMUUD_params.num_perclass,GMUUD_params.data_prune_path,device)
        select_dataset = torch.load(GMUUD_params.data_prune_path)
        remaining_dataloader = torch.utils.data.DataLoader(select_dataset, GMUUD_params.batch_size, shuffle=True)


    Student_model = GMUUD_Unlearning(GMUUD_params,Teacher1_model,Teacher2_model,Student_model,remaining_dataloader,unlearning_dataloader,optimizer,device)
    Acc_eval(Student_model, unlearning_dataloader,remaining_dataloader, test_dataloader,device)

    # Save model
    # path = "./Model/Student/Student_model_{}_{}_{}_{}.pt".format(Student_model_name,GMUUD_params.KD_epochs, GMUUD_params.seed_num, GMUUD_params.data_name)
    # torch.save({
    #     'model_state_dict': Student_model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, path)




if __name__=='__main__':
    GMUUD_Unlearning_main()


    
        

