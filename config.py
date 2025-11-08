
import time

train_data=  
test_data=  
log_dir =  


train_files = [[]]

val_files = [[]]


checkpoint = "" 



grad_checkpoint = False  
 
patch_size = (32,32,1024)
max_input_size = int(8192*16)
min_input_size = max_input_size

 
vocab_size = 256+3


 
device = 6  
epochs = 20
lr = 0.00004  
warmup = 100
batch_size = 1
 
weight_decay = 3e-3
workers = 8
 
limit_tokens = max_input_size

iter_val = 70
iter_print = 10
iter_save = 70*5
iter_break_val = 500

 
seed = int(time.time())%120

