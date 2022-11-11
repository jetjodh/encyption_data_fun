import pandas as pd
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchmetrics as plm
from datasets import ClassLabel
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer# Load the BERT tokenizer.
from sklearn.model_selection import StratifiedKFold
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm as tqdm
import wandb
from transformers import BertForSequenceClassification,RobertaForSequenceClassification, AdamW, BertConfig

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
class Memes(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, aug=None, text_aug=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.aug = aug
        self.text_aug = text_aug
        
    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.file.iloc[idx, 1]
        labeler = ClassLabel(num_classes=11, names=["AES","AES128","AES256","DES3","Blowfish","Camellia","CAST5","ChaCha20", "RC4","IDEA","SEED"])
        label = labeler.str2int(self.file.iloc[idx, 1])
        label = np.array([label])
        text = self.file.iloc[idx, 0]
        if self.text_aug:
            text = self.text_aug.augment(text)
        encoded_sent = tokenizer.encode(
             text, # Sentence to encode.
             add_special_tokens = True, # Add '[CLS]' and '[SEP]' # This function also supports truncation and conversion
             # to pytorch tensors, but we need to do padding, so we
             # can't use these features :( .
             #max_length = 128, # Truncate all sentences.
             #return_tensors = 'pt', # Return pytorch tensors.
             )
        # I've chosen 64 somewhat arbitrarily. It's slightly larger than the
        # maximum training sentence length of 47...
        MAX_LEN = 256
        #print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
        #print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))# Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        encoded_sent = pad_sequences([encoded_sent], maxlen=MAX_LEN, dtype="long", 
         value=0, truncating="post", padding="post")
        #print('\Done.')
        attention_masks=[]
        #print(encoded_sent)
        att_mask = [int(token_id > 0) for token_id in encoded_sent[0]]
        attention_masks.append(att_mask)
        encoded_sent = torch.tensor(encoded_sent)
        attention_masks = torch.tensor(attention_masks)
        label = torch.tensor(label)
        sample = {'text':encoded_sent[0],'mask':attention_masks[0],'label': label}
        
        return sample

class Test(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file = csv_file
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        text = self.file.iloc[idx, 0]
        encoded_sent = tokenizer.encode(
             text, # Sentence to encode.
             add_special_tokens = True, # Add '[CLS]' and '[SEP]' # This function also supports truncation and conversion
             # to pytorch tensors, but we need to do padding, so we
             # can't use these features :( .
             #max_length = 128, # Truncate all sentences.
             #return_tensors = 'pt', # Return pytorch tensors.
             )
        # I've chosen 64 somewhat arbitrarily. It's slightly larger than the
        # maximum training sentence length of 47...
        MAX_LEN = 256
        #print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
        #print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))# Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        encoded_sent = pad_sequences([encoded_sent], maxlen=MAX_LEN, dtype="long", 
         value=0, truncating="post", padding="post")
        #print('\Done.')
        attention_masks=[]
        #print(encoded_sent)
        att_mask = [int(token_id > 0) for token_id in encoded_sent[0]]
        attention_masks.append(att_mask)
        encoded_sent = torch.tensor(encoded_sent)
        attention_masks = torch.tensor(attention_masks)
        sample = {'text':encoded_sent[0],'mask':attention_masks[0]}
        
        return sample

skf = StratifiedKFold(n_splits=3)
data = pd.read_csv('encrypted_data.csv')
y = np.zeros(len(data))
for train_index, test_index  in skf.split(data,y):
    X_train, X_test = data.to_numpy()[train_index], data.to_numpy()[test_index]


train = pd.DataFrame(data=X_train, columns=data.columns)
val = pd.DataFrame(data=X_test, columns=data.columns)
train = Memes(train, root_dir='')
val = Memes(val,root_dir='')
# test = Test(data[test_index],root_dir='')

dataset_sizes = {'train':len(train), 'val':len(train)}
train = DataLoader(train, batch_size=1,
                        shuffle=True)
val = DataLoader(train, batch_size=1,
                        shuffle=True)
# test = DataLoader(test, batch_size=32,
#                         shuffle=False)
dataloaders = {'train':train,'val':val}

wandb.init(project="encrypted_data")

device = 'cuda'

def send_my_epochs(epoch,losses,aucs):
    return {'Epoch':epoch, 'Epoch loss': losses, 'Epoch AUC':aucs}
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    met1 = plm.Accuracy().to("cuda")
    met2 = plm.Precision().to("cuda")
    met3 = plm.Recall().to("cuda")
    cm = plm.ConfusionMatrix(num_classes=11)
    wandb.watch(model)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_roc = 0.0
    best_acc = 0.0
    best_prec = 0.0
    best_recall = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0
            running_prec = 0.0
            running_recall = 0.0
            runnin_roc = 0.0
            running_labels = []
            running_preds = []
            
            tk0 = tqdm(dataloaders[phase], total=dataset_sizes[phase]//dataloaders[phase].batch_size +1)
            counter = 0
            # Iterate over data.
            for i,sample in enumerate(tk0,0):                
                texts = sample['text']
                offsets = sample['mask']
                labels = sample['label']
                texts = texts.to(device='cuda')
                offsets = offsets.to(device='cuda')
                labels = labels.to(device='cuda')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(texts,offsets)
                    # print(labels.shape)
                    # print(outputs.shape)            
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels.squeeze(1))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    acc = met1(preds, labels.squeeze(1))
                    prec = met2(preds, labels)
                    recall = met3(preds, labels)
                # statistics
                running_loss += loss.item() * texts.size(0)
                running_acc += acc.item()
                running_prec += prec.item()
                running_recall += recall.item()
                running_labels.extend(labels.squeeze(1).cuda())
                running_preds.extend(preds.cuda())
                counter += 1
                tk0.set_postfix(loss=(running_loss / (counter * dataloaders[phase].batch_size)))
                #print(_)
                #print(preds)
                #print(labels.squeeze(1))
            if phase == 'train':
                scheduler.step()
            if phase == 'val':
                print(cm(np.array(running_preds),np.array(running_labels)))   

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]
            epoch_prec = running_prec / dataset_sizes[phase]
            epoch_recall = running_recall / dataset_sizes[phase]
            epoch_roc = roc_auc_score(running_labels,running_preds)

            if phase == 'train':
                wandb.log({"Train ROC_AUC": epoch_roc, "Train Loss": epoch_loss})
                
            if phase == 'val':
                wandb.log({"Test ROC_AUC": epoch_roc, "Test Loss": epoch_loss})
                    
            print('{} Loss: {:.4f} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_prec, epoch_recall, epoch_roc))
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_prec = epoch_prec
                best_recall = epoch_recall
                best_roc = epoch_roc
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best val AUC: {:4f}'.format(best_roc))
    print('Best val accuracy: {:4f}'.format(best_acc))
    print('Best val recall: {:4f}'.format(best_recall))
    print('Best val prec: {:4f}'.format(epoch_prec))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # torch.save(model,'bert_dec.pt')
    return {'loss':best_loss, 'AUC':best_roc,'model':model}

import torch.nn.functional as F
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
bert = BertForSequenceClassification.from_pretrained(
 "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
 num_labels = 11, # The number of output labels--2 for binary classification.
 # You can increase this for multi-class tasks. 
 output_attentions = False, # Whether the model returns attentions weights.
 output_hidden_states = False, # Whether the model returns all hidden-states.
)
# roberta = RobertaForSequenceClassification.from_pretrained('roberta-base',
# num_labels = 2, # The number of output labels--2 for binary classification.
#  # You can increase this for multi-class tasks. 
# output_attentions = False, # Whether the model returns attentions weights.
# output_hidden_states = False, # Whether the model returns all hidden-states.
# )
# model_ft = models.wide_resnet50_2(pretrained=True)
# del bert.classifier
# print(bert)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features2 = bert
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self,x2, x3):
        x2 = x2.to(torch.int64)
       
        x2 = self.features2(x2,x3)[0]
        x = self.softmax(x2)
        return x


model = MyModel()
torch.cuda.empty_cache()
model.to('cuda')
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = AdamW(model.parameters(), lr= 4.539374136412813e-05, weight_decay= 0.09937039516365159, eps= 2.238637604422851e-07)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)

import time
#print(model)
hist = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=15)