import torch
from MedicalNet.model import generate_model  # This is an example; the actual import may differ

def get_med3dmodell(model_type:str="resnet_10_23dataset",input_dim:tuple = (112,112,6)):
    # https://github.com/Tencent/MedicalNet
    class options():
        def __init__(self,model_type,input_dim):
            if ("18" in model_type) or ("34" in model_type):
                self.resnet_shortcut = 'A'
            else:
                self.resnet_shortcut = 'B'
            self.model = "resnet"
            self.n_seg_classes = 1
            self.n_epochs = 1
            self.no_cuda = True
            self.pretrain_path = ''
            self.num_workers = 0
            self.model_depth = 10 
            self.input_D = input_dim[2]
            self.input_H = input_dim[1]
            self.input_W = input_dim[0]
            self.phase = "train"
            self.pretrain_path = f"./MedicalNet/pretrain/{model_type}.pth"
            self.new_layer_names = []
        
    
    opt = options(model_type,input_dim)
    # Initialize the model
    model = generate_model(opt)

    # check for any unknown layer
    pretrain_dict = torch.load(f"./MedicalNet/pretrain/{model_type}.pth",map_location=torch.device('cpu'))['state_dict']
    pretrain_dict = {key.replace("module.", ""): value for key, value in pretrain_dict.items()}
    model_dict = model[0].state_dict()
    for key in pretrain_dict.keys():
        if key not in model_dict:
            print(f"Key {key} not in model_dict")

    return model[0]