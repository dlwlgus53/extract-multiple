import os
import pdb
import json
import nltk
import torch
import argparse
import ontology
from dataset import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW
from knockknock import email_sender

import datetime
now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser()

parser.add_argument('--patience' ,  type = int, default=3)
parser.add_argument('--batch_size' , type = int, default=8)
parser.add_argument('--base_trained_model', type = str, default = 'bert-base-uncased', help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--gpu_number' , type = int,  default = 0, help = 'which GPU will you use?')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--log_file' , type = str,  default = f'logs/log_{now_time}.txt',)
parser.add_argument('--test_path' ,  type = str,  default = '../data/MultiWOZ_2.1/test_data.json')
parser.add_argument('--value_list_path' ,  type = str,  default = '../data/MultiWOZ_2.1/value_list.json')

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
    makedirs("./logs");
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained_model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_trained_model)
    test_raw = json.load(open(args.test_path, "r"))
    test_dataset = Dataset(args.test_path, 'test', tokenizer, False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    log_file = open(args.log_file, 'w')
    device = torch.device(f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    torch.cuda.empty_cache()

    print("use trained model")
    log_file.write("use trained model")
    # model.load_state_dict(torch.load(args.pretrained_model))
    
    log_file.write(str(args))
    model.to(device)

    collect_predict = get_predict(model, test_loader, device, tokenizer,log_file)
    # collect predict 에서 전에 turn에 있는 belief 그대로 가져오는 코드
    
    evaluate(test_raw, collect_predict)
    
    
def evaluate(collect_answer, collect_predict):
    # joint goal accuracy
    
    joint_goal_acc = 0
    joint_goal_count = 0
    
    
    slot_acc = 0
    slot_count =0 
    
    for dial_key in collect_predict.keys():
        ans_dial, pred_dial = collect_answer[dial_key]['log'], collect_predict[dial_key]
        current_belief = {}
        pdb.set_trace()
        for turn_key in pred_dial.keys():
            ans_belief , pred_belief = ans_dial[turn_key]['belief'], pred_dial[turn_key]
            
            joint_goal_count +=1
            slot_count += len(ans_belief)
            diff = ans_belief - pred_belief
            
            if len(diff) == 0: # 전부 다 맞은 경우
                joint_goal_acc +=1
                slot_acc += len(ans_belief)
            
            else: 
                slot_acc += (len(ans_belief) - len(diff))
                
    return joint_goal_acc/joint_goal_count, slot_acc/slot_count


def edit_distance_compenstae(schema, pred_text,value_list):
    candidate = value_list[schema]
    candidate_score = [nltk.edit_distance(pred_text, value) for value in candidate ]
    index  = candidate_score.index(min(candidate_score))
    return candidate[index]
    
    

def get_predict(model, test_loader, device, tokenizer, log_file):
    value_list = json.load(open(args.value_list_path, "r"))
    model.eval()
    collect_predict =  defaultdict(lambda: defaultdict(dict))
    with torch.no_grad():
        test_loader = tqdm(test_loader)
        for iter,batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            dial_id = batch['dial_id']
            turn_id = batch['turn_id'].tolist()
            schema = batch['schema']
            
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

            pred_start_positions = torch.argmax(outputs['start_logits'], dim=1).to('cpu')
            pred_end_positions = torch.argmax(outputs['end_logits'], dim=1).to('cpu')
            for b in range(len(batch)):
                ans_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][start_positions[b]:end_positions[b]+1]))
                pred_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][pred_start_positions[b]:pred_end_positions[b]+1]))
                pred_text = edit_distance_compenstae(batch['schema'][b], pred_text,value_list)
                collect_predict[dial_id[b]][turn_id[b]][schema[b]] = pred_text
            if iter%100 ==0:
                log_file.write(f"ans text : {ans_text}\n pred_text : {pred_text}\n")        
                break     
    return collect_predict
    
if __name__ =="__main__":
    main()

    