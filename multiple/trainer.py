import torch
from tqdm import tqdm
import gc
import pdb 

from sklearn.metrics import accuracy_score
def train(model, train_loader, optimizer, device):
        model.train()
        loss_sum = 0
        t_train_loader = tqdm(train_loader)
        anss, preds  = [] , []
        ACC =0
        for iter, batch in enumerate(t_train_loader):
             
            anss += batch['labels'].to('cpu').tolist()
            optimizer.zero_grad()
            batch = {k:v.to(device)for k, v in batch.items()}
            outputs = model(input_ids = batch['input_ids'], token_type_ids = batch['token_type_ids'],\
                             attention_mask=batch['attention_mask'], labels = batch['labels'])
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            t_train_loader.set_description("Loss %.04f ACC %.04f" % (loss, ACC))
            preds += torch.max(outputs[1], axis = 1).indices.to('cpu').tolist()
            
        
            if iter %100 == 0 and len(anss) != 0:
                ACC = accuracy_score(anss, preds)
                anss , preds = [] , []
        
        gc.collect()
        torch.cuda.empty_cache()




def valid(model, dev_loader, device, tokenizer, log_file):

    model.eval()
    anss = []
    preds = []
    loss_sum = 0
    print("Validation start")
    with torch.no_grad():
        log_file.write("\n")
        t_dev_loader = tqdm(dev_loader)
        for iter,batch in enumerate(t_dev_loader):
            anss += batch['labels'].tolist()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, labels = labels)
            loss_sum += outputs[0].to('cpu')
            preds += torch.max(outputs[1], axis = 1).indices.to('cpu').tolist()
            
            t_dev_loader.set_description("Loss %.04f  | step %d" % (outputs[0].to('cpu'), iter))

        torch.cuda.empty_cache()
           
    return  anss, preds, loss_sum.item()/iter
        
        