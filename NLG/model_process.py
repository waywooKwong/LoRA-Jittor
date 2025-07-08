import torch
state_dict = torch.load('pretrained_model/gpt2-medium-pytorch_model.bin', map_location='cpu')
torch.save(state_dict, 'pretrained_model/gpt2-medium-pytorch_model_zip.bin', _use_new_zipfile_serialization=True)