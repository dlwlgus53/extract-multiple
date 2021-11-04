import os
import pdb
import json
import nltk
import torch
from tqdm import tqdm
from collections import defaultdict

value_list_path = '../data/MultiWOZ_2.1/value_list.json'


def evaluate(collect_answer, belief_e, belief_m):
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
                
    return joint_goal_acc/joint_goal_count, slot_acc/slot_count, final_file


def edit_distance_compenstae(schema, pred_text,value_list):
    candidate = value_list[schema]
    candidate_score = [nltk.edit_distance(pred_text, value) for value in candidate ]
    index  = candidate_score.index(min(candidate_score))
    return candidate[index]
    
    
def get_predict(model, test_loader, device, tokenizer, type, log_file):
    collect_predict =  defaultdict(lambda: defaultdict(dict))
    if type == 'extract':
        value_list = json.load(open(value_list_path, "r"))
    
    model.eval()
    with torch.no_grad():
        test_loader = tqdm(test_loader)
        for iter,batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dial_id = batch['dial_id']
            turn_id = batch['turn_id'].tolist()
            schema = batch['schema']
            
            
            if type == 'extract':
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
            else:
                labels =  batch['labels'].to(device) # TODO
                token_type_ids = batch['token_type_ids'].to(device)
            
            if type == 'extract':
                outputs = model(input_ids = input_ids, attention_mask = attention_mask, start_positions = start_positions, end_positions = end_positions)
                pred_start_positions = torch.argmax(outputs['start_logits'], dim=1).to('cpu')
                pred_end_positions = torch.argmax(outputs['end_logits'], dim=1).to('cpu')
            else:
                outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, labels = labels)
                pred_index = torch.max(outputs[1], axis = 1).indices.to('cpu').tolist()


            for b in range(len(batch)):
                
                if type == 'extract':
                    ans_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][start_positions[b]:end_positions[b]+1]))
                    pred_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][pred_start_positions[b]:pred_end_positions[b]+1]))
                    pred_text = edit_distance_compenstae(batch['schema'][b], pred_text,value_list)
                    collect_predict[dial_id[b]][turn_id[b]][schema[b]] = pred_text
                    
                    if iter%100 ==0:
                        log_file.write(f"ans text : {ans_text}\n pred_text : {pred_text}\n")        
                    
                    
                else:
                    ans_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][start_positions[b]:end_positions[b]+1]))
                    pred_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][pred_start_positions[b]:pred_end_positions[b]+1]))
                    pred_text = edit_distance_compenstae(batch['schema'][b], pred_text,value_list) # 번호 순서에 맞게!
                    collect_predict[dial_id[b]][turn_id[b]][schema[b]] = pred_text
                    
                    if iter%100 ==0:
                        log_file.write(f"ans text : {ans_text}\n pred_text : {pred_text}\n")        
                      
    return collect_predict


    