import os
import pdb
import json
import nltk
import torch
from tqdm import tqdm
from collections import defaultdict
import ontology
import editdistance

value_list_path = './data/MultiWOZ_2.1/value_list.json'



def _attentionTo3(belief):
    attention = ['attraction','hotel','restaurant']
    
    belief = {k:belief[k] for k in belief if k.split('-')[0] in attention}
    return belief
def evaluate(answer_path, extract_belief, multiple_belief):
    final_belief =  defaultdict(lambda: defaultdict(dict))
    
    answer = json.load(open(answer_path, "r"))
    
    joint_goal_acc = 0
    joint_goal_count = 0
    
    slot_acc = 0
    slot_count =0 
    
    for dial_key in answer.keys():
        ans_dial, extract_dial, multiple_dial = answer[dial_key]['log'], extract_belief[dial_key], multiple_belief[dial_key]
        current_belief = {}
        for turn_key in range(len(ans_dial)):
            ans_turn_belief, extract_turn_belief, multiple_turn_belief = ans_dial[turn_key]['belief'], extract_dial[turn_key], multiple_dial[turn_key]
            ans_turn_belief = _attentionTo3(ans_turn_belief)

            joint_goal_count +=1
            slot_count += len(ans_turn_belief)
            
            merged = _attentionTo3({**extract_turn_belief,**multiple_turn_belief})
            current_belief.update(merged)
            
            shared_items = {schema : ans_turn_belief[schema] for schema in ans_turn_belief if schema in current_belief and ans_turn_belief[schema] == current_belief[schema]}
            
            if len(ans_turn_belief) == len(shared_items): # 전부 다 맞은 경우
                joint_goal_acc +=1
                slot_acc += len(ans_turn_belief)
            
            else: 
                slot_acc += len(shared_items)
        
            final_belief[dial_key][turn_key] = current_belief
                
    return joint_goal_acc/joint_goal_count, slot_acc/slot_count, final_belief


def edit_distance_compenstae(schema, pred_text,value_list):
    candidate = value_list[schema] + ['[CLS]']
    
    candidate_score = [editdistance.eval(pred_text, value) for value in candidate ]
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
            else: # multiple
                labels =  batch['labels'].to(device) # TODO
                token_type_ids = batch['token_type_ids'].to(device)
            
            if type == 'extract':
                outputs = model(input_ids = input_ids, attention_mask = attention_mask, start_positions = start_positions, end_positions = end_positions)
                pred_start_positions = torch.argmax(outputs['start_logits'], dim=1).to('cpu')
                pred_end_positions = torch.argmax(outputs['end_logits'], dim=1).to('cpu')
            else: # multiple
                outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, labels = labels)
                pred_index = torch.max(outputs[1], axis = 1).indices.to('cpu').tolist()


            for b in range(len(dial_id)): # refactor
                
                if type == 'extract':
                    ans_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][start_positions[b]:end_positions[b]+1]))
                    pred_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[b][pred_start_positions[b]:pred_end_positions[b]+1]))
                    pred_text = edit_distance_compenstae(batch['schema'][b], pred_text,value_list)
                    if pred_text != '[CLS]':
                        collect_predict[dial_id[b]][turn_id[b]][schema[b]] = pred_text
                    
                    if iter%100 ==0:
                        log_file.write(f"ans text : {ans_text}\n pred_text : {pred_text}\n")        
                    
                    
                else:
                    
                    index = pred_index[b]
                    label = labels[b]
                    ans_text = ontology.QA[schema[b]]['values'][label]
                    try:
                        pred_text = ontology.QA[schema[b]]['values'][index]
                    except IndexError as e:
                        pred_text = ontology.QA[schema[b]]['values'][-1]
                        
                        
                    collect_predict[dial_id[b]][turn_id[b]][schema[b]] = pred_text
                    
                    if iter%100 ==0:
                        log_file.write(f"ans text : {ans_text}\n pred_text : {pred_text}\n")        
                      
    return collect_predict


    