import torch  
import numpy as np  
import random

from model import BBT
import config as cfg
import math
import time
from utils import load_pretrained_weights
  
 
def set_seed(seed=42):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  

    torch.backends.cudnn.deterministic = True    
    torch.backends.cudnn.benchmark = False  
    
set_seed(42)  

model = BBT()

model = load_pretrained_weights(model,'/xxxxx.pth')

device = 6
model = model.cuda(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")




def read_bytes(content):
            
    bytes = bytearray(content, 'utf-8')
    bytes = [byte for byte in bytes]
    
    return bytes

def convert_tokens_to_string(tokens):

        while tokens:
            try:
                tokens = tokens.decode("utf-8")
                return tokens
      
            except UnicodeDecodeError:

                tokens = tokens[:-1]
        return ""  



content = """100 + 12 ="""


content = """This is a jet. Next, I will provide you with all the particle information contained within this jet 
following the format below: Index: Electric charge of the particle (charge), Energy of the particle (energy), 
Three components of momentum (Px, Py, Pz), Logarithm of the particle's energy (log10(energy)), 
Logarithm of the particle's transverse momentum (log10(pt)), Difference in pseudorapidity between the particle 
and the jet axis (Delta eta), Difference in azimuthal angle between the particle and the jet axis (Delta phi), 
Logarithm of the particle's Pt relative to the jet Pt (logptrel), Logarithm of the particle's energy relative 
to the jet energy (logerel), Angular separation between the particle and the jet axis (Delta R), Transverse 
impact parameter of the track (d0), Uncertainty associated with the measurement of the d0 (d0err), Longitudinal 
impact parameter of the track (z0), Uncertainty associated with the measurement of the z0 (z0err), What's the particle
 type of this particle (one of the following types: electron, muon, charged kaon, charged pion, proton, neutral hadron, 
 or photon). 0: -1, 39.997673, (39.071407, 2.615292, -8.147392), 1.602035, 1.592830, -0.008121, -0.011592, -0.364432, 
 -0.366413, 0.014154, -0.040276, 0.042930, -0.041573, 0.045784, charged pion. 1: 1, 20.167332, (19.597847, 1.669760, -4.454019),
  1.304648, 1.293779, 0.009850, 0.006567, -0.663483, -0.663800, 0.011838, 0.028787, 0.044804, -0.049005, 0.046869, charged pion. 
  2: -1, 9.883578, (9.615731, 1.313970, -1.864669), 0.994914, 0.987000, -0.023738, 0.057378, -0.970262, -0.973534, 0.062095, 
  -0.045781, 0.047454, 0.041122, 0.048744, charged pion. 3: 0, 8.805975, (8.579162, 0.373816, -1.950241), 0.944777, 0.933857, 
  0.010491, -0.034884, -1.023405, -1.023671, 0.036427, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 
  4: -1, 4.309661, (4.009282, 0.373691, -1.529662), 0.634443, 0.604945, 0.156585, 0.014509, -1.352317, -1.334005, 0.157256, 
  0.026351, 0.051953, -0.044425, 0.052641, charged pion. 5: 1, 2.258724, (2.168113, 0.496886, -0.367296), 0.353863, 0.347198, 
  -0.050323, 0.146860, -1.610064, -1.614585, 0.155242, -0.047911, 0.056039, 0.040003, 0.056180, charged pion. 6: 0, 1.877179,
   (1.827124, 0.040637, -0.428684), 0.273506, 0.261875, 0.017756, -0.056192, -1.695387, -1.694942, 0.058930, 0.000000, 0.000000,
    0.000000, 0.000000, photon. 7: 0, 1.619287, (1.607403, -0.168118, -0.100421), 0.209324, 0.208487, -0.152614, -0.182640, -1.748775,
     -1.759124, 0.238009, 0.000000, 0.000000, 0.000000, 0.000000, photon. 8: 1, 1.053021, (0.983063, 0.322094, -0.139196), 
     0.022437, 0.014724, -0.080555, 0.238192, -1.942538, -1.946011, 0.251445, 0.081441, 0.064434, -0.087411, 0.064554, 
     charged pion. 9: 1, 0.938284, (0.906876, 0.017885, -0.195732), -0.027666, -0.042368, -0.000560, -0.058710, 
     -1.999630, -1.996114, 0.058713, -0.049936, 0.067066, -0.043708, 0.067504, charged pion. 10: 0, 0.360000, (0.344053, 0.040322, -0.097987), 
     -0.443698, -0.460412, 0.064515, 0.038235, -2.417674, -2.412146, 0.074994, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 11: 
     0, 0.301790, (0.289375, 0.032678, -0.079192), -0.520296, -0.535788, 0.053985, 0.034021, -2.493050, -2.488744, 0.063811, 0.000000, 
     0.000000, 0.000000, 0.000000, neutral hadron. 12: 0, 0.273339, (0.266608, -0.037073, -0.047543), -0.563298, -0.569968, -0.038989, 
     -0.216596, -2.527231, -2.531746, 0.220077, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 13: 0, 0.272749, 
     (0.267844, 0.021731, -0.046682), -0.564237, -0.570693, -0.041854, 0.002527, -2.527956, -2.532686, 0.041930, 0.000000, 
     0.000000, 0.000000, 0.000000, neutral hadron. 14: 0, 0.253881, (0.252464, -0.022352, -0.014757), -0.595371, -0.596105, 
     -0.156516, -0.166733, -2.553367, -2.563819, 0.228686, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 15: 0, 0.250365,
      (0.246755, 0.040264, -0.013166), -0.601426, -0.602027, -0.162073, 0.083319, -2.559289, -2.569874, 0.182236, 0.000000, 0.000000,
       0.000000, 0.000000, photon. 16: 0, 0.199311, (0.189398, -0.043531, -0.044256), -0.700468, -0.711447, 0.011096, -0.304344, 
       -2.668709, -2.668916, 0.304546, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 17: 0, 0.069126, (0.065918, 0.003825, 
       -0.020461), -1.160358, -1.180268, 0.090413, -0.020464, -3.137530, -3.128806, 0.092700, 0.000000, 0.000000, 0.000000, 0.000000, 
       neutral hadron. 18: 0, 0.050117, (0.045704, 0.009277, -0.018352), -1.300016, -1.331280, 0.169301, 0.121826, -3.288542, -3.268464, 0.208577,
        0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 19: 0, 0.035876, (0.011056, 0.000865, -0.034119), -1.445201, -1.955083, 1.627693, 
        -0.000322, -3.912345, -3.413649, 1.627693, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 20: 0, 0.015259,
        (0.004150, -0.001361, -0.014621), -1.816465, -2.359733, 1.708215, -0.395278, -4.316995, -3.784914, 1.753352, 0.000000, 0.000000, 
        0.000000, 0.000000, neutral hadron. class: """




print(content)
pred_byte = None
with torch.no_grad():
    model.eval()
    print("test=====================")
    src_bytes = read_bytes(content)
    head = [256]
    res = []
    for i in range(1000):
             
        input_bytes = torch.tensor(src_bytes+res,dtype=torch.long)
        
        input_bytes = input_bytes.unsqueeze(0)
                 
        input_id = input_bytes.cuda(device)

        output = model(input_id)


        _, predicted = torch.max(output, -1)
        pred_byte = predicted[0,-1].item()


        if pred_byte>255:
            break
        res.append(pred_byte)
        
        pred_txt = convert_tokens_to_string(bytes(res))
        if pred_txt.rfind("<|im_end|>")>-1:
                break


    pred_txt = convert_tokens_to_string(bytes(res))
    print(f"{content} ==> {pred_txt}")




            




