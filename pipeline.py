from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
from IPython import embed
import sys
import pickle
import time
import os

import torch.nn.functional as F

load_data = True
load_data_file = './dataset.pkl'

if SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    sys.path.append(SDK_PATH)

import mmsdk
import os
import re
import numpy as np
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError

# create folders for storing the data
if not os.path.exists(DATA_PATH):
    check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

# download highlevel features, low-level (raw) data and labels for the dataset MOSI
# if the files are already present, instead of downloading it you just load it yourself.
# here we use CMU_MOSI dataset as example.

DATASET = md.cmu_mosi

try:
    md.mmdataset(DATASET.highlevel, DATA_PATH)
except RuntimeError:
    print("High-level features have been downloaded previously.")

try:
    md.mmdataset(DATASET.raw, DATA_PATH)
except RuntimeError:
    print("Raw data have been downloaded previously.")
    
try:
    md.mmdataset(DATASET.labels, DATA_PATH)
except RuntimeError:
    print("Labels have been downloaded previously.")

# list the directory contents... let's see what features there are
data_files = os.listdir(DATA_PATH)
print('\n'.join(data_files))


# define your different modalities - refer to the filenames of the CSD files
visual_field = 'CMU_MOSI_VisualFacet_4.1'
acoustic_field = 'CMU_MOSI_COVAREP'
text_field = 'CMU_MOSI_ModifiedTimestampedWords'
label_field = 'CMU_MOSI_Opinion_Labels'

if load_data and os.path.exists(load_data_file):
    with open(load_data_file, 'rb') as f:
        dataset = pickle.load(f)

else:

    features = [
        text_field, 
        visual_field, 
        acoustic_field
    ]
    
    recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
    dataset = md.mmdataset(recipe)
    
    # Take a look into the dataset
    print(list(dataset.keys()))
    print("=" * 80)
    
    print(list(dataset[visual_field].keys())[:10])
    print("=" * 80)
    
    some_id = list(dataset[visual_field].keys())[15]
    print(list(dataset[visual_field][some_id].keys()))
    print("=" * 80)
    
    print(list(dataset[visual_field][some_id]['intervals'].shape))
    print("=" * 80)
    
    print(list(dataset[visual_field][some_id]['features'].shape))
    print(list(dataset[text_field][some_id]['features'].shape))
    print(list(dataset[acoustic_field][some_id]['features'].shape))
    print("Different modalities have different number of time steps!")
    
    # Align
    # we define a simple averaging function that does not depend on intervals
    def avg(intervals: np.array, features: np.array) -> np.array:
        try:
            return np.average(features, axis=0)
        except:
            return features
    
    # first we align to words with averaging, collapse_function receives a list of functions
    dataset.align(text_field, collapse_functions=[avg])
    
    
    # Add labels
    
    # we add and align to lables to obtain labeled segments
    # this time we don't apply collapse functions so that the temporal sequences are preserved
    label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
    dataset.add_computational_sequences(label_recipe, destination=None)
    dataset.align(label_field)
    # check out what the keys look like now
    print(list(dataset[text_field].keys())[55])

    # Save the dataset.
    with open(load_data_file, 'wb') as f:
        pickle.dump(dataset, f)
    
# Split the dataset.
# obtain the train/dev/test splits - these splits are based on video IDs
train_split = DATASET.standard_folds.standard_train_fold
dev_split = DATASET.standard_folds.standard_valid_fold
test_split = DATASET.standard_folds.standard_test_fold

# inspect the splits: they only contain video IDs
print(test_split)

# we can see they are in the format of 'video_id[segment_no]', but the splits was specified with video_id only
# we need to use regex or something to match the video IDs...
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook
from collections import defaultdict

# a sentinel epsilon for safe division, without it we will replace illegal values with a constant
EPS = 0

# construct a word2id mapping that automatically takes increment when new words are encountered
all_words = []
for segment in dataset[label_field].keys():
    all_words.append(dataset[text_field][segment]['features'][:])
all_words= np.concatenate(all_words, 0).squeeze(1)
vocabs = list(set(all_words))
vocabs = [v.decode('utf-8') for v in vocabs]
vocabs += ['<unk>', '<pad>']

## Load pretrained embeddings
# define a function that loads data from GloVe-like embedding files
# we will add tutorials for loading contextualized embeddings later
# 2196017 is the vocab size of GloVe here.

def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    f = open(path_to_embedding, 'r')
    found = 0
    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()

def return_unk():
    return UNK

if os.path.exists(CACHE_PATH):
    pretrained_emb, word2id = torch.load(CACHE_PATH)
else:
    word2id = defaultdict(lambda: len(word2id))
    cnt = 0
    for v in vocabs:
        word2id[v] = cnt
        cnt += 1

    # turn off the word2id - define a named function here to allow for pickling
    word2id.default_factory = return_unk

    if WORD_EMB_PATH is not None:
        pretrained_emb = load_emb(word2id, WORD_EMB_PATH)
        torch.save((pretrained_emb, word2id), CACHE_PATH)

    else:
        pretrained_emb = None


# place holders for the final train/dev/test dataset
train = []
dev = []
test = []

# define a regular expression to extract the video ID out of the keys
pattern = re.compile('(.*)\[.*\]')
num_drop = 0 # a counter to count how many data points went into some processing issues

UNK = word2id['<unk>']
PAD = word2id['<pad>']

for segment in dataset[label_field].keys():
    
    # get the video ID and the features out of the aligned dataset
    vid = re.search(pattern, segment).group(1)
    label = dataset[label_field][segment]['features']
    _words = dataset[text_field][segment]['features']
    _visual = dataset[visual_field][segment]['features']
    _acoustic = dataset[acoustic_field][segment]['features']

    # if the sequences are not same length after alignment, there must be some problem with some modalities
    # we should drop it or inspect the data again
    if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
        print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
        num_drop += 1
        continue

    # remove nan values
    label = np.nan_to_num(label)
    _visual = np.nan_to_num(_visual)
    _acoustic = np.nan_to_num(_acoustic)

    # remove speech pause tokens - this is in general helpful
    # we should remove speech pauses and corresponding visual/acoustic features together
    # otherwise modalities would no longer be aligned
    words = []
    visual = []
    acoustic = []
    for i, word in enumerate(_words):
        if word[0] != b'sp':
            words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
            visual.append(_visual[i, :])
            acoustic.append(_acoustic[i, :])

    words = np.asarray(words)
    visual = np.asarray(visual)
    acoustic = np.asarray(acoustic)

    # z-normalization per instance and remove nan/infs
    visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
    acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

    if vid in train_split:
        train.append(((words, visual, acoustic), label, segment))
    elif vid in dev_split:
        dev.append(((words, visual, acoustic), label, segment))
    elif vid in test_split:
        test.append(((words, visual, acoustic), label, segment))
    else:
        print(f"Found video that doesn't belong to any splits: {vid}")

print(f"Total number of {num_drop} datapoints have been dropped.")



# let's see the size of each set and shape of data
print(len(train))
print(len(dev))
print(len(test))

print(train[0][0][1].shape)
print(train[0][1].shape)
print(train[0][1])

print(f"Total vocab size: {len(word2id)}")

# Collate
def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, lengths

# construct dataloaders, dev and test could use around ~X3 times batch size since no_grad is used during eval
batch_sz = 56 #FIXME: in tutorials is 56
train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)

# let's create a temporary dataloader just to see how the batch looks like
temp_loader = iter(DataLoader(test, shuffle=True, batch_size=8, collate_fn=multi_collate))
batch = next(temp_loader)

print(batch[0].shape) # word vectors, padded to maxlen
print(batch[1].shape) # visual features
print(batch[2].shape) # acoustic features
print(batch[3]) # labels
print(batch[4]) # lengths

# Let's actually inspect the transcripts to ensure it's correct
id2word = {v:k for k, v in word2id.items()}
examine_target = train
idx = np.random.randint(0, len(examine_target))
print(' '.join(list(map(lambda x: id2word[x], examine_target[idx][0][0].tolist()))))
# print(' '.join(examine_target[idx][0]))
print(examine_target[idx][1])
print(examine_target[idx][2])

## Define multimodal models.
# let's define a simple model that can deal with multimodal variable length sequence

class DNN(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(DNN, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3 


class OneLayerLSTM(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text. 
    * Comments: In tfn codes, the inputs are batch_size first (batch_size, sequence_len, in_size) and lstm is implemented directly on padded sequence. here our inputs are (sequence_len, batch_size, in_size) and lstm is imiplemented on packed sequence.
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(OneLayerLSTM, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional) #  batch_size=False
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        Args:
            x: tensor of shape (sequence_len, batch_size, in_size). The order is different from the source code in TFN, and we use packed sequence rather than directly padded sequence as inputs.
        '''       
        packed_sequence = pack_padded_sequence(x, lengths)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

class Fusion(nn.Module):
    '''
    Implement fusion.
    '''
    def __init__(self, input_dims, hidden_dims, text_out, dropouts, post_fusion_dim, output_size=1, fusion_type='simple'):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
            output_size - int, specifying the number of classes. Here a continuous prediction is produced which will be used for binary classification
            fusion_type - 'simple': simple concatenation and fully connected (in tutorials); 'tfn': tensor fusion network fusion.
        '''
        super(Fusion, self).__init__()
       
        self.fusion_type = fusion_type
 
        # dimensions are specified
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.post_fusion_dim = post_fusion_dim 
        self.post_fusion_prob = dropouts[3]

        self.embed = nn.Embedding(len(word2id), input_sizes[2])

        self.output_size = output_size

        # define sub-networks
        self.audio_dropout = dropouts[0] 
        self.audio_subnet = DNN(self.audio_in, self.audio_hidden, self.audio_dropout)


        self.video_dropout = dropouts[1]
        self.video_subnet = DNN(self.video_in, self.video_hidden, self.video_dropout)


        self.text_out = text_out
        self.text_dropout = dropouts[2]
        self.text_subnet = OneLayerLSTM(self.text_in, self.text_hidden, self.text_out, dropout=self.text_dropout)

        self.audio_out_dim = self.audio_hidden
        self.video_out_dim = self.video_hidden
        self.text_out_dim = self.text_out

        # define the post fusion layers
        if self.fusion_type == 'simple':

            self.fc1 = nn.Linear(sum((self.audio_out_dim, self.video_out_dim, self.text_out_dim)), self.post_fusion_dim) 
            self.fc2 = nn.Linear(self.post_fusion_dim, output_size)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(self.post_fusion_prob) 
            self.bn = nn.BatchNorm1d(sum((self.audio_out_dim, self.video_out_dim, self.text_out_dim)))

        elif self.fusion_type == 'tfn':
            self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)


            self.post_fusion_layer_1 = nn.Linear((self.text_out_dim + 1) * (self.video_out_dim + 1) * (self.audio_out_dim + 1), self.post_fusion_dim)
            self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
            self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_size)

        else:
            raise ValueError('fusion_type can only be simple or tfn!')


    def forward(self, audio_x, video_x, text_x, lengths_x):
        '''
        Args:
            audio_x - a tensor with shape (max_len, batch_size, in)
            video_x - same as audio_x
            text_x - same as audio_x
            lengths_x - a tensor with the length of batch_size, containing the exact lengths of features
        '''
        audio_x = torch.mean(audio_x, dim=0, keepdim=True).squeeze() # audio are averaged along time axis. (max_len, batch_size, in_dim) -> (batch_size, in_dim)
        audio_h = self.audio_subnet(audio_x)

        video_x = torch.mean(video_x, dim=0, keepdim=True).squeeze() # videos are averaged along time axis
        video_h = self.video_subnet(video_x)        

        text_x = self.embed(text_x)
        text_h = self.text_subnet(text_x, lengths_x) 
        batch_size = lengths_x.size(0)

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = torch.FloatTensor([6]).type(DTYPE)
        self.output_shift = torch.FloatTensor([-3]).type(DTYPE)

        if self.fusion_type == 'simple':
            h = torch.cat((text_h, video_h, audio_h), dim=1)
            h = self.bn(h)
            h = self.fc1(h)
            h = self.dropout(h)
            h = self.relu(h)
            o = self.fc2(h)
        elif self.fusion_type == 'tfn':
           # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
            _audio_h = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), audio_h), dim=1)
            _video_h = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), video_h), dim=1)
            _text_h = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), text_h), dim=1)

            # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
            # we want to perform outer product between the two batch, hence we unsqueenze them to get
            # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
            # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
            fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
            
            # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
            # we have to reshape the fusion tensor during the computation
            # in the end we don't keep the 3-D tensor, instead we flatten it
            fusion_tensor = fusion_tensor.view(-1, (self.audio_out_dim + 1) * (self.video_out_dim + 1), 1) 
            fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)
    
            post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
            post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
            post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
            post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
            o = post_fusion_y_3 * self.output_range + self.output_shift
        else:
            raise ValueError('Fusion_type can only be specified as simple or tfn!')        
        return o




## Train a model
from tqdm import tqdm_notebook
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

CUDA = torch.cuda.is_available()

text_size = 300
visual_size = 47
acoustic_size = 74

# define some model settings and hyper-parameters
fusion_type = 'tfn' #FIXME
MAX_EPOCH = 1000

input_sizes = [acoustic_size, visual_size, text_size]
hidden_sizes = (4, 16, 128)
text_out = 64
dropouts = (0.3, 0.3, 0.3, 0.3)
post_fusion_dim = 32
output_size = 1 # Output a continuous value

dropout = 0.25
output_size = 1
curr_patience = patience = 8
num_trials = 3
grad_clip_value = 1.0
weight_decay = 0.1


# TODO: specify the save name
date = time.strftime('%m%d%H%M%S', time.localtime())
save_dir = './ckpt'
if not os.path.exists(save_dir): os.makedirs(save_dir)

model_save_path = os.path.join(save_dir, '{}_model.std'.format(date))
optim_save_path = os.path.join(save_dir, '{}_optim.std'.format(date) )

model = Fusion(input_sizes, hidden_sizes, text_out, dropouts, post_fusion_dim, output_size, fusion_type=fusion_type)
if pretrained_emb is not None:
    model.embed.weight.data = pretrained_emb
model.embed.requires_grad = False
optimizer = Adam([param for param in model.parameters() if param.requires_grad], weight_decay=weight_decay)

if CUDA:
    model.cuda()
criterion = nn.L1Loss(reduction='sum')
criterion_test = nn.L1Loss(reduction='sum')
best_valid_loss = float('inf')
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
lr_scheduler.step() # for some reason it seems the StepLR needs to be stepped once first
train_losses = []
valid_losses = []
for e in range(MAX_EPOCH):
    model.train()
    train_iter = tqdm_notebook(train_loader)
    train_loss = 0.0
    for batch in train_iter:
        model.zero_grad()
        t, v, a, y, l = batch
        batch_size = t.size(0)
        if CUDA:
            t = t.cuda()
            v = v.cuda()
            a = a.cuda()
            y = y.cuda()
            l = l.cuda()
        y_tilde = model(a, v, t, l)
        loss = criterion(y_tilde, y)
        loss.backward()
        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
        optimizer.step()
        train_iter.set_description(f"Epoch {e}/{MAX_EPOCH}, current batch loss: {round(loss.item()/batch_size, 4)}")
        train_loss += loss.item()
    train_loss = train_loss / len(train)
    train_losses.append(train_loss)
    print(f"Training loss: {round(train_loss, 4)}")

    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        for batch in dev_loader:
            model.zero_grad()
            t, v, a, y, l = batch
            if CUDA:
                t = t.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            y_tilde = model(a, v, t, l)
            loss = criterion(y_tilde, y)
            valid_loss += loss.item()
    
    valid_loss = valid_loss/len(dev)
    valid_losses.append(valid_loss)
    print(f"Validation loss: {round(valid_loss, 4)}")
    print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
    if valid_loss <= best_valid_loss:
        best_valid_loss = valid_loss
        print("Found new best model on dev set!")
        torch.save(model.state_dict(), model_save_path)
        torch.save(optimizer.state_dict(), optim_save_path)
        curr_patience = patience
    else:
        curr_patience -= 1
        if curr_patience <= -1:
            print("Running out of patience, loading previous best model.")
            num_trials -= 1
            curr_patience = patience
            model.load_state_dict(torch.load(model_save_path))
            optimizer.load_state_dict(torch.load(optim_save_path))
            lr_scheduler.step()
            print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
    
    if num_trials <= 0:
        print("Running out of patience, early stopping.")
        break

model.load_state_dict(torch.load(model_save_path))
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for batch in test_loader:
        model.zero_grad()
        t, v, a, y, l = batch
        if CUDA:
            t = t.cuda()
            v = v.cuda()
            a = a.cuda()
            y = y.cuda()
            l = l.cuda()
        y_tilde = model(a, v, t, l)
        loss = criterion_test(y_tilde, y)
        y_true.append(y_tilde.detach().cpu().numpy())
        y_pred.append(y.detach().cpu().numpy())
        test_loss += loss.item()
print(f"Test set performance: {test_loss/len(test)}")
y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)
                  
y_true_bin = y_true >= 0
y_pred_bin = y_pred >= 0
bin_acc = accuracy_score(y_true_bin, y_pred_bin)
print(f"Test set accuracy is {bin_acc}")

