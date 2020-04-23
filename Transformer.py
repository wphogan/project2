# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_handler import *

class TransformerModel(nn.Module):

    def __init__(self, ntoken, emb_size, nhead, nhid, nlayers):
        """
        emb_size: Embedding Size for the input
        ntoken: Number of tokens Vocab Size
        nhead: Number of transformer heads in the encoder
        nhid: Number of hidden units in transformer encoder layer
        nlayer: Number of layers in transformer encoder
        """
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
        self.emb_size = emb_size
        self.ntoken = ntoken
        self.nhead = nhead
        self.nhid = nhid
        self.nlayer = nlayers
        
        self.ninp = emb_size ### NEED TO CHECK THIS
        """
        1. Initialize position input embedding, position encoding layers
        2. Initialize transformer encoder with nlayers and each layer 
        having nhead heads and nhid hidden units.
        3. Decoder can be implemented directly on top of the encoder as a linear layer. 
           To keep things simple, we are predicting one token for each of the input tokens. 
           We can pad the input to have the same length as target to ensure we can generate all target tokens. 
           You may experiment with a transformer decoder and use teacher forcing during training, 
           but it is not necessary to do so.
        """
        # 1
        # ninp == emb_size
        self.pos_encoder = PositionalEncoding(emb_size)
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid)
        
        # 2
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, emb_size)
        
        #3
        self.decoder = nn.Linear(emb_size, ntoken)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src):
        """
        src: tensor of shape (seq_len, batch_size)
        
        Returns:
            output: tensor of shape (seq_len, batch_size, vocab_size)
        """
        
        """
        1. Embed the source sequences and add the positional encoding.
        2. Pass the sequence to the transformer encoder
        3. Generate and return scores using decoder linear layer
        """
        # 1: Embed the source sequences and add the positional encoding.
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        
        # 2: Pass the sequence to the transformer encoder 
        output = self.transformer_encoder(src)
        
        # 3: Generate and return scores using decoder linear layer
        output = self.decoder(output)
        
        return output

class PositionalEncoding(nn.Module):
    """
    Adds positional embedding to the input for conditioning on time. 
    This is already implemented for you, but you can try other variants of positional encoding.
    Read the paper "Attention is all you need" for more details on this. 
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: tensor of shape (seq_len, batch_size, embedding_size)
        Returns:
            x: tensor of shape (seq_len, batch_size, embedding_size)
        """
        x = x + self.pe[:x.size(0), :]
        return x

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def convert_dataset_to_tensor(data_pairs, max_len):
    source_rows = []
    target_rows = []
    for pair in data_pairs:
        source, target = pair
        
        source_index_list = string_to_index_list(source, char_to_index, end_token)
        source_index_list = source_index_list + [end_token for et in range(max_len - (len(source_index_list)))]
        
        target_index_list = [start_token] + string_to_index_list(target, char_to_index, end_token)
        target_index_list = target_index_list + [end_token for et in range(max_len - (len(target_index_list)))]
        
        source_rows.append( torch.LongTensor(source_index_list) )
        target_rows.append( torch.LongTensor(target_index_list) )
        
    source_tensors = torch.stack(source_rows)
    target_tensors = torch.stack(target_rows)

    return source_tensors, target_tensors

# def evaluate(model, val_data):
def evaluate(model, val_inputs,val_targets):
    # Feel free the change the arguments this function accepts
    model.eval() # Turn on the evaluation mode
    val_inputs  = val_inputs.T
    output      = model(val_inputs)
    val_targets_one_dim = val_targets.reshape(-1)
    val_loss    = criterion(output.view(-1, ntokens), val_targets_one_dim)
    # Return validation loss and
    # Accuracy -> percentage of validation sequences that were translated correctly.
    
    
    ### ADD ACCURACY 
    max_args_out = torch.argmax(output, dim=2)
    temp = torch.eq(val_inputs, max_args_out)
    total_right = (temp == True).sum()
    total_wrong = (temp == False).sum()
    val_accuracy = total_right.item() / (total_right.item() + total_wrong.item()) #### TEMPORARY
    
    model.train() # Turn off the evaluation mode, turn on training mode
    return val_loss/max_len, val_accuracy


def translate(model, input_sequence):
    gen_string = []
    index_to_char = idx_dict['index_to_char']
    
    # Translates the input sequence to piglatin using the trained model
    model.eval()
    
    # Convert input_sequence to a tensor of appropriate shape
    words = input_sequence.split()
    for word in words: 
        gen_word = []
        word_idx = string_to_index_list(word, char_to_index, end_token)
        word_idx = word_idx + [end_token for et in range(max_len - (len(word_idx)))]
        word_tensor = torch.LongTensor(word_idx).unsqueeze(1)
        
        # process it through the model and predict the translation.
        output = model(word_tensor)
        new_tokens = torch.argmax(output.squeeze(1), dim=1)
        for token in new_tokens:
            ni = index_to_char[token.item()]
            if token.item() == start_token: continue #SOS
            elif token.item() != end_token:
                 ni = index_to_char[token.item()]
                 gen_word.append(ni)
            else: break #EOS
        gen_string.append(str(''.join(str(x) for x in gen_word)))

    translation = ' '.join(gen_string)
        
    model.train()
    return translation
    
# +
line_pairs, vocab_size, idx_dict = load_data()

char_to_index = idx_dict['char_to_index']
start_token = idx_dict['start_token']
end_token = idx_dict['end_token']

num_lines = len(line_pairs)
num_train = int(0.8 * num_lines)
train_pairs, val_pairs = line_pairs[:num_train], line_pairs[num_train:]

# Create source and target tensors for the pairs: 
# Shape of source (NumExamples, MaxSentLength), Shape of target (NumExamples, MaxSentLength). 

# Make sure source and targets have the same shape by padding them to the same length with end tokens. 
# This is because our transformer implementation predicts one token for each input token
# During inference for a single example, we can sufficiently pad the input with end tokens.

line_pairs, vocab_size, idx_dict = load_data()

num_lines = len(line_pairs)
num_train = int(0.8 * num_lines)
train_pairs, val_pairs = line_pairs[:num_train], line_pairs[num_train:]

source_strings = [pair[0] for pair in line_pairs]
target_strings = [pair[1] for pair in line_pairs]

max_input_len = max([ len(source_string)+1 for source_string in source_strings])
max_target_len = max([ len(target_string)+2 for target_string in target_strings])
max_len = max(max_input_len, max_target_len)



train_inputs, train_targets = convert_dataset_to_tensor(train_pairs, max_len)
val_inputs, val_targets = convert_dataset_to_tensor(val_pairs, max_len)

print ("Train Sequences", train_inputs.size(), train_targets.size())
print ("Val Sequences", val_inputs.size(), val_targets.size())
# -

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = vocab_size # the size of vocabulary
batch_size = 16
emsize = 50 # embedding dimension
nhid = 50 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
lr = 1 # 1.0 # learning rate
epochs = 50 # The number of epochs
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers).to(device)
print("lr:{}, bs:{}, epochs:{}".format(lr, batch_size,epochs))
# +
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# Implement Training Loop Here
model.train()

for epoch in range(epochs):
    epoch_losses = []
    
    num_tensors = len(train_pairs)
    num_batches = int(np.ceil(num_tensors / float(batch_size)))
                
    for batch_no in range(num_batches):
        # input, target = get_batch(batch_no, .. )  
        start = batch_no * batch_size
        end = start + batch_size

        inputs = train_inputs[start:end].permute(1,0)
        targets = train_targets[start:end].permute(1,0)
        
        # Process input to the model, get the ouput, compute loss
        optimizer.zero_grad()
        output = model(inputs)
        output = output.view(-1, ntokens)
        targets = targets.reshape(-1)

        # backpropagate loss and update weights
        loss = criterion(output, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        
        # Calculate avg training loss over the batches.
        epoch_losses.append(loss.item())
        
        
    train_loss = np.mean(epoch_losses)
    val_loss, val_acc = evaluate(model, val_inputs, val_targets)
    
    sample_translation = translate(model, "i love deep learning")
    print ("Epoch:{} | Train Loss:{} | Val Loss:{} | Val Acc:{} ".format(epoch, train_loss, val_loss, val_acc))
    print (sample_translation)
    model.train() 
    scheduler.step()


