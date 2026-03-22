import torch

def get_device():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            if any(x in name.lower() for x in ['nvidia','rtx','geforce','gtx']):
                return i
    return 'cpu'
