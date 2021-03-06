import os
import json
import torch
import datetime
import argparse
from tqdm import tqdm
from knockknock import email_sender
from test_utils import get_predict, evaluate
from torch.utils.data import DataLoader
from extract.dataset import Dataset as Dataset_e
from multiple.dataset import Dataset as Dataset_m
from transformers import AutoModelForQuestionAnswering, AutoModelForMultipleChoice, AutoTokenizer

now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--base_trained_model', type = str, default = 'bert-base-uncased', help =" pretrainned model from 🤗")
parser.add_argument('--gpu_number' , type = int,  default = 0, help = 'which GPU will you use?')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--log_file' , type = str,  default = f'logs/log_{now_time}.txt',)
parser.add_argument('--test_path' ,  type = str,  default = './data/MultiWOZ_2.1/sample_data.json')
parser.add_argument('--extract_model_path' , type = str,  default = './extract/model/woz.pt')
parser.add_argument('--multiple_model_path' , type = str,  default = './multiple/model/woz.pt')
parser.add_argument('--extract_batch_size', type = int, default= 16)
parser.add_argument('--multiple_batch_size', type = int, default = 8)
parser.add_argument('--max_options' , type = int,  default = 9, help = 'max number of options')



args = parser.parse_args()

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       

args = parser.parse_args()

# @email_sender(recipient_emails=["jihyunlee@postech.ac.kr"], sender_email="knowing.deep.clean.water@gmail.com")
def main():
    makedirs("./data"); makedirs("./logs"); 
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained_model, use_fast=True)
    
    model_e = AutoModelForQuestionAnswering.from_pretrained(args.base_trained_model)
    model_m = AutoModelForMultipleChoice.from_pretrained(args.base_trained_model)
    
    test_e_dataset = Dataset_e(args.test_path, 'sample', tokenizer, False)
    test_e_loader = DataLoader(test_e_dataset, args.extract_batch_size, shuffle=False)
    
    test_m_dataset = Dataset_m(args.test_path, 'sample', tokenizer, args.max_options, False)
    test_m_loader = DataLoader(test_m_dataset, args.multiple_batch_size, shuffle=False)
    
    log_file = open(args.log_file, 'w')
    device = torch.device(f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    torch.cuda.empty_cache()

    print("Load models")
    
    model_e.load_state_dict(torch.load(args.extract_model_path))
    model_m.load_state_dict(torch.load(args.multiple_model_path))
    
    
    log_file.write(str(args))
    model_e.to(device)
    model_m.to(device)
    
    
    belief_e = get_predict(model_e, test_e_loader, device, tokenizer, 'extract', log_file)
    belief_m = get_predict(model_m, test_m_loader, device, tokenizer, 'multiple', log_file)
    
    joint_goal_acc, slot_acc, final_belief = evaluate(args.test_path, belief_e, belief_m)
    print('JGA : ', str(joint_goal_acc*100), 'Slot Acc : ', str(slot_acc * 100))
    # save the final_file
    
    with open( "belief.json", "w") as f:
        json.dump(final_belief, f, indent=2)
        
    return {'joint_goal_acc' : joint_goal_acc, 'slot_acc' : slot_acc}

    
if __name__ =="__main__":
    main()

    