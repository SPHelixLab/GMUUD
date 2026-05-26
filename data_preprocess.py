import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
import os
import random
import torch.nn.functional as F

def seed_torch(seed = 600):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_datasets(data_name,data_dir):

    if data_name == 'cifar10':
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),  # data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalization
        ])

        # 数据集预处理
        train_datasets = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
        test_datasets = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    elif data_name == 'cifar100':

        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        train_datasets = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform) # no augmentation
        test_datasets = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
    elif data_name == 'mnist':

        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        train_datasets = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform) # no augmentation
        test_datasets = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)


    return train_datasets, test_datasets

def get_dataloader(train_datasets, test_datasets,batch_size):
    def worker_init_fn(worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    # 数据分批次
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True,
                                                   worker_init_fn=worker_init_fn, num_workers=0)
    # 从测试集中挑选前256个样本作为Df
    Dp_size = batch_size
    indices_all = [i for i in range(len(test_datasets))]
    #print(indices_all)
    # 随机挑选256个只要每次的种子相同就相同
    #indices_selest = np.random.choice(len(test_datasets), Dp_size, replace=False)
    indices_select = [i for i in range(Dp_size)]
    #print(indices_select)
    indices_new = np.delete(indices_all, indices_select)
    test_subset = torch.utils.data.Subset(test_datasets, indices_select)
    Dp_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=256, shuffle=False)
    test_new = torch.utils.data.Subset(test_datasets, indices_new)
    test_dataloader = torch.utils.data.DataLoader(test_new, batch_size=batch_size, shuffle=True,
                                                   worker_init_fn=worker_init_fn, num_workers=0)
    DandDp = torch.utils.data.ConcatDataset([train_datasets, test_subset])
    DandDp_dataloader = torch.utils.data.DataLoader(DandDp, batch_size=256, shuffle=True,
                                                    worker_init_fn=worker_init_fn, num_workers=0)
    return train_dataloader, test_dataloader, Dp_dataloader, DandDp_dataloader




def train(model, loader, device, optimizer, criterion):
    model.train()
    train_loss = 0.
    train_corrects = 0
    for i, (inputs, labels) in enumerate(tqdm(loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)

        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)
    epoch_loss = train_loss / len(loader.dataset)
    epoch_acc = train_corrects / len(loader.dataset) * 100.
    return epoch_loss, epoch_acc




def test(model, loader, criterion, device):
    model.eval()
    with torch.no_grad():
        test_corrects = 0
        test_loss = 0.
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            loss = criterion(logits, labels)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
        epoch_loss = test_loss / len(loader.dataset)
        epoch_acc = test_corrects / len(loader.dataset) * 100.
    return epoch_loss, epoch_acc

def dataset_prune(dataset,num_classes,model,num_perclass,path,device):
    # 构建一个二维数组，用于存储每个类别的样本数据
    class_data = [[] for cls in range(num_classes)]
    # 遍历整个训练集，将每个样本添加到相应的类别列表中
    for img, label in dataset:
        class_data[label].append(img)
    print('*'*100)
    print('each class selects {} samples \n'.format(num_perclass))
    dataset_select = []
    for cls in range(num_classes):
        # 将新的数据集封装为 PyTorch Dataset 对象
        class_cls_dataset = torch.utils.data.TensorDataset(torch.stack(class_data[cls]),
                                                        torch.tensor([cls for i in range(len(class_data[cls]))]))
        print('class_cls_dataset:', len(class_cls_dataset))
        # class_cls_dataloader = torch.utils.data.DataLoader(class_cls_dataset, batch_size=256, shuffle=False)

        # 获取对应类别样本在Warmupmodel上的重要分数
        model.eval()
        EL2N_score = []
        with torch.no_grad():
            for img, label in tqdm(class_cls_dataset):
                img = torch.unsqueeze(img, dim=0).to(device)
                losits = model(img)
                Warmup_outputs = F.softmax(losits, dim=None, _stacklevel=3, dtype=None)
                class_c_onehot = F.one_hot(torch.tensor(cls), num_classes).to(device)
                EL2N_score.append(torch.norm(Warmup_outputs - class_c_onehot, p=2, dim=1))

        # 按重要分数从大到小排序获得排序后对应原来的索引
        sorted_id_c = sorted(range(len(EL2N_score)), key=lambda k: EL2N_score[k], reverse=True)
        # 取最大的num_percls样本
        select_id_c = sorted_id_c[:num_perclass]
        # 取出数据集中对应下标的样本
        class_cls_dataset_select = torch.utils.data.Subset(class_cls_dataset, select_id_c)
        print('Class {} selects {} samples'.format(cls, len(class_cls_dataset_select)))

        dataset_select.append(class_cls_dataset_select)

    dataset_select_allclass = torch.utils.data.ConcatDataset(dataset_select)
    print('total select {} samples'.format(len(dataset_select_allclass)))

    torch.save(dataset_select_allclass, path)
