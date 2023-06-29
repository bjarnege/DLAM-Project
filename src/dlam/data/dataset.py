import medmnist
import torch
import numpy as np
import torchvision.transforms 

def load_dataset(dataset_name: str, split: str, download_dataset: bool = True):
    info = medmnist.INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    return DataClass(split=split, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[.5], std=[.5])
    ]), download=download_dataset)

def load_concat_dataset(split: str,
                        dataset_names: list = ["organamnist", "organcmnist", "organsmnist"],
                        download_dataset: bool = True,
                        transform: bool = True):
    output_dataclass = {"imgs": np.empty((0, 28, 28)),
                        "labels": np.empty((0, 1)),
                        "infos": [],
                        "split": split}

    for dataset_name in dataset_names:
        info = medmnist.INFO[dataset_name]
        DataClass = getattr(medmnist, info["python_class"])
        dataclass = DataClass(split=split, download=download_dataset)
        output_dataclass["imgs"] = np.concatenate((dataclass.imgs, output_dataclass["imgs"]))
        output_dataclass["labels"] = np.concatenate((dataclass.labels, output_dataclass["labels"]))
        output_dataclass["infos"].append(info)


    # normalize and convert to tensor
    if transform == True:
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[.5], std=[.5])])

        output_dataclass["imgs"] = transforms(output_dataclass["imgs"]).permute((1, 0, 2)).contiguous()
        output_dataclass["labels"] = torch.Tensor(output_dataclass["labels"])


    return output_dataclass
