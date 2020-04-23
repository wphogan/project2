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
import os
import pdb
import argparse
import pickle as pkl

from collections import defaultdict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Local imports
import utils
import data_handler
from data_handler import *
import attn_vis

# Prevent OMP Error #15 ("Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# +
# Options/ Hyperparameters required to train and test the model
opts = EasyDict()

opts.n_epochs = 100
opts.batch_size = 16
opts.learning_rate = 0.0015
opts.lr_decay = 0.99
opts.hidden_layer_size = 12
opts.model_name = "attention_rnn"
opts.checkpoints_dir = "./checkpoints/"+opts.model_name 
opts.function_f_type = 'cosine' # 'cosine', 'pairwise', 'neural_net', 'neural_net_deep'

print(opts)

TEST_SENTENCE = 'we love deep learning'
TEST_WORD_ATTENTION = 'learning'
VOWEL_TEST_WORD_ATTENTION = 'universe'
HYPHEN_TEST_WORD_ATTENTION = 'south-easterly'
# -

utils.create_dir_if_not_exists(opts.checkpoints_dir)

line_pairs, vocab_size, idx_dict = load_data()

# dividing the line pairs into 8:2, train and val split
num_lines = len(line_pairs)
num_train = int(0.8 * num_lines)
train_pairs, val_pairs = line_pairs[:num_train], line_pairs[num_train:]

# +
train_dict = create_dict(train_pairs)
val_dict = create_dict(val_pairs)

# Study the structure of the created train_dict and val_dict variables

def equal_shape(in_tensor, in_shape):
    """
    Takes in tensor and list of shape dims, returns true/false
    """
    return in_tensor.shape == torch.Size(in_shape)
# +
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, opts):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.opts = opts

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, inputs):
        """Forward pass of the encoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """

        batch_size, seq_len = inputs.size()
        hidden = self.init_hidden(batch_size)
        cell = self.init_hidden(batch_size)
        
        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        assert equal_shape(encoded, [batch_size, seq_len, hidden.shape[1]])
        
        annotations = []

        for i in range(seq_len):
            x = encoded[:,i,:]  # Get the current time step, across the whole batch
            hidden, cell = self.lstm_cell(x, (hidden, cell) )
            annotations.append(hidden)

        annotations = torch.stack(annotations, dim=1)
        return annotations, hidden, cell

    def init_hidden(self, bs):
        """Creates a tensor of zeros to represent the initial hidden states
        of a batch of sequences.

        Arguments:
            bs: The batch size for the initial hidden state.

        Returns:
            hidden: An initial hidden state of all zeros. (batch_size x hidden_size)
        """
        return Variable(torch.zeros(bs, self.hidden_size))


class Attention(nn.Module):
    def __init__(self, hidden_size, opts):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.f_type      = opts.function_f_type
        self.relu        = nn.ReLU()
        self.softmax     = nn.Softmax(dim=1)
        # ------------
        # FILL THIS IN
        # ------------
    
        # Create a two layer fully-connected network. 
        # [[hidden_size*2 --> hidden_size], [ReLU], [hidden_size --> 1]]
        
        # Default attention neural net
        self.attention_network = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )    
        
        # Deeper attention neural net
        self.attention_network_deep = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )
        
        # Distance functions
        self.cosine = nn.CosineSimilarity()
        self.pairwise = nn.PairwiseDistance()

    def cosine_simularity(self,x,y, seq):
        cos = []
        for i in range(seq):
            cos.append(self.cosine(x[:,i,:], y[:,i,:]))
        cos = torch.stack(cos)
        cos = self.relu(cos).T.unsqueeze(2) # bs x seq x 1
        return cos
    
    def pairwise_distance(self,x,y, seq):
        pair = []
        for i in range(seq):
            pair.append(self.pairwise(x[:,i,:], y[:,i,:]))
        pair = torch.stack(pair)
        pair = self.relu(pair).T.unsqueeze(2) # bs x seq x 1
        return pair

    def forward(self, hidden, annotations):
        """The forward pass of the attention mechanism.

        Arguments:
            hidden: The current decoder hidden state. (batch_size x hidden_size)
            annotations: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            output: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)
            The output must be a softmax weighting over the seq_len annotations.
        """

        batch_size, seq_len, hid_size = annotations.size()
        expanded_hidden = hidden.unsqueeze(1).expand_as(annotations)

        # ------------
        # FILL THIS IN
        # ------------

        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        if self.f_type == 'cosine':
            unnormalized_attention = self.cosine_simularity(expanded_hidden,annotations, seq_len)
        
        elif self.f_type == 'pairwise':
            unnormalized_attention = self.pairwise_distance(expanded_hidden,annotations, seq_len)
        
        else: 
            concat = torch.cat((expanded_hidden,annotations),2)
            assert equal_shape(concat, [batch_size, seq_len, hid_size * 2])
            
            #reshaped_for_attention_net = concat.reshape(1,0,2) # batch x seq_len x hidden_size*2
            reshaped_for_attention_net = concat # in: bs x seq x hidden_size*2 // out: bs x seq x hidden_size*2
            assert equal_shape(reshaped_for_attention_net, [batch_size, seq_len, hid_size * 2])
            
            if self.f_type == 'neural_net_deep':
                attention_net_output = self.attention_network_deep(reshaped_for_attention_net) # bs x seq x hidden*2
            else: 
                attention_net_output = self.attention_network(reshaped_for_attention_net) # bs x seq x hidden*2
                
            assert equal_shape(attention_net_output, [batch_size, seq_len, 1])
            
            # unnormalized_attention = attention_net_output.reshape(1,0,2)  # Reshape attention net output to have dimension batch_size x seq_len x 1
            unnormalized_attention = attention_net_output #in: bs x seq x 1 // out: bs x seq x 1
            assert equal_shape(unnormalized_attention, [batch_size, seq_len, 1])

        return self.softmax(unnormalized_attention)
        #return unnormalized_attention


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, opts):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.lstm_cell = nn.LSTMCell(input_size=hidden_size*2, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=hidden_size, opts=opts)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.soft_max = nn.Softmax(dim=1)
        
    def softmax(self,x):
        return self.soft_max(x)

    def forward(self, x, h_prev, c_prev, annotations):
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            x: Input token indexes across a batch for a single time step. (batch_size x 1)
            h_prev: The hidden states from the previous step, across a batch. (batch_size x hidden_size)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch. (batch_size x vocab_size)
            h_new: The new hidden states, across a batch. (batch_size x hidden_size)
            attention_weights: The weights applied to the encoder annotations, across a batch. (batch_size x encoder_seq_len x 1)
        """
        batch_size, seq_len, hid_size = annotations.size()
        embed = self.embedding(x)    # batch_size x 1 x hidden_size
        embed = embed.squeeze(1)     # batch_size x hidden_size
        assert equal_shape(embed, [batch_size, hid_size])
        # ------------
        # FILL THIS IN
        # ------------
        # attention_weights = ...
        # context = ...
        # embed_and_context = ...
        # h_new, c_new = ...
        # output = ...
        
        ####### NEED TO CHECK ANNOTATIONS HERE
        attention_weights = self.attention(h_prev, annotations)
        attention_weights = attention_weights.squeeze(2)
        assert equal_shape(attention_weights, [batch_size, seq_len])
        
        ####### NEED TO CHECK THIS:
        a = attention_weights.permute(1,0)
        b = annotations.permute(2,1,0)
        attn_cross_annotate =  a * b
        
        context = torch.sum(attn_cross_annotate,dim=1) # should result in batch x hidden_size
        context = context.T
        assert equal_shape(context, [batch_size, hid_size])
        
        embed_and_context = torch.cat((embed,context),1) # should result in shape: batch x hidden_size*2
        assert equal_shape(embed_and_context, [batch_size, hid_size*2])
        
        h_new, c_new = self.lstm_cell(embed_and_context, (h_prev, c_prev)) # LSTM Input: input, (h_0, c_0)
        
        output = self.out(h_new) #in:  batch x hidden  // out: batch x vocab
        assert equal_shape(output, [batch_size, vocab_size])
        
        return output, h_new, c_new, attention_weights


# -

##########################################################################
### Setup: Create Encoder, Decoder Objects ###
##########################################################################
encoder = Encoder(vocab_size=vocab_size, hidden_size=opts.hidden_layer_size, opts = opts)
decoder = AttentionDecoder(vocab_size=vocab_size, hidden_size=opts.hidden_layer_size, opts=opts)

def train_model(train_dict, val_dict, idx_dict, encoder, decoder, opts):
    """Runs the main training loop; evaluates the model on the val set every epoch.
        * Prints training and val loss each epoch.
        * Prints qualitative translation results each epoch using TEST_SENTENCE

    Arguments:
        train_dict: The training word pairs, organized by source and target lengths.
        val_dict: The validation word pairs, organized by source and target lengths.
        idx_dict: Contains char-to-index and index-to-char mappings, and start & end token indexes.
        encoder: An encoder model to produce annotations for each step of the input sequence.
        decoder: A decoder model to generate output tokens.
        opts: The input arguments for hyper-parameters and others.
    """
    
    # Define your loss function and optimizers
    # ....
    # parameters = list(encoder.parameters()) + list(decoder.parameters())
    parameters = [
        {'params': encoder.parameters(), 'lr': opts.learning_rate},
        {'params': decoder.parameters(), 'lr': opts.learning_rate}
    ]
    optimizer  = optim.Adam(parameters, lr=opts.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_token   = idx_dict['start_token']
    end_token     = idx_dict['end_token']
    char_to_index = idx_dict['char_to_index']

    loss_log = open(os.path.join(opts.checkpoints_dir, 'loss_log.txt'), 'w')

    best_val_loss = 1e6
    train_losses,val_losses = [],[]

    for epoch in range(opts.n_epochs):
        
        # decay the learning rate of the optimizer
        # ....
        optimizer.param_groups[0]['lr'] *= opts.lr_decay

        epoch_losses = []

        for key in train_dict:

            input_strings, target_strings = zip(*train_dict[key])
            
            # Make your input tensor and the target tensors
            # HINT : use the function string_to_index_list given in data_handler.py
            # input_tensors = ....
            # output_tensors = ....
            input_tensors, output_tensors = generate_input_tensors(char_to_index, end_token, input_strings, target_strings)
            
            num_tensors = len(input_tensors)
            num_batches = int(np.ceil(num_tensors / float(opts.batch_size)))
            
            optimizer.zero_grad()
            for i in range(num_batches):

                start = i * opts.batch_size
                end = start + opts.batch_size
                
        

                # Define inputs and targets for THIS batch, beginning at index 'start' to 'end'
                # inputs = ....
                # outputs = ....
                inputs = Variable(input_tensors[start:end])
                targets = Variable(output_tensors[start:end])
                
                # The batch size may be different in each epoch
                BS = inputs.size(0)

                encoder_annotations, encoder_hidden, encoder_cell = encoder.forward(inputs)

                # The last hidden state of the encoder becomes the first hidden state of the decoder
                # decoder_hidden = ....
                # decoder_cell = .. either zeros, or last encoder hidden state
                decoder_hidden = Variable(encoder_hidden)
                decoder_cell = Variable(encoder_cell)
                
                # Define the first decoder input. This would essentially be the start_token
                # decoder_input = ....
                decoder_input = Variable(torch.LongTensor([start_token]*BS).unsqueeze(1))
                loss = 0.0

                seq_len = targets.size(1)  # Gets seq_len from BS x seq_len

                for i in range(seq_len):
                    decoder_output, decoder_hidden, decoder_cell, attention_weights = decoder.forward(decoder_input, \
                                                                                        decoder_hidden, \
                                                                                        decoder_cell, \
                                                                                        encoder_annotations)

                    current_target = targets[:,i]
                    assert equal_shape(current_target, [BS])
                    
                    # Calculate the cross entropy between the decoder distribution and Ground truth (current_target)
                    # loss += ....
                    loss = loss + criterion(decoder_output, current_target)
                    
                    # Find out the most probable character (ni) from the softmax distribution produced
                    # ni = ....

                    decoder_input = Variable(targets[:,i].unsqueeze(1))
                    assert equal_shape(decoder_input, [BS,1])

                loss /= float(seq_len)
                epoch_losses.append(loss.item())

                # Compute gradients
                loss.backward()

                # Update the parameters of the encoder and decoder
                optimizer.step()

        train_loss = np.mean(epoch_losses)
        val_loss = evaluate(val_dict, encoder, decoder, idx_dict, criterion, opts)

        if val_loss < best_val_loss:
            utils.store_checkpoints(encoder, decoder, idx_dict, opts)
        
        # Visualize attention
        attn_vis.visualize_attention(TEST_WORD_ATTENTION,
                                      encoder,
                                      decoder,
                                      idx_dict,
                                      opts,
                                      save=os.path.join(opts.checkpoints_dir,\
                                                        'train_attns/attn-epoch-{}.png'.format(epoch)))
        if epoch == 99:
            attn_vis.visualize_attention(HYPHEN_TEST_WORD_ATTENTION,
                                      encoder,
                                      decoder,
                                      idx_dict,
                                      opts,
                                      save=os.path.join(opts.checkpoints_dir,\
                                                        'train_attns/vowel-attn-epoch-{}.png'.format(epoch)))
            attn_vis.visualize_attention(VOWEL_TEST_WORD_ATTENTION,
                                      encoder,
                                      decoder,
                                      idx_dict,
                                      opts,
                                      save=os.path.join(opts.checkpoints_dir,\
                                                        'train_attns/hyphen-attn-epoch-{}.png'.format(epoch)))
            
        gen_string = find_pig_latin(TEST_SENTENCE, encoder, decoder, idx_dict, opts)

        print("Epoch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f} | Gen: {:20s}".format(epoch, train_loss, val_loss, gen_string))

        loss_log.write('{} {} {}\n'.format(epoch, train_loss, val_loss))
        loss_log.flush()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        utils.store_loss_plots(train_losses, val_losses, opts)



def evaluate(data_dict, encoder, decoder, idx_dict, criterion, opts):
    """Evaluates the model on a held-out validation or test set. 
    This should be pretty straight-forward if you have figured out how to do the training correctly.
    From then, it's just copy and paste.

    Arguments:
        data_dict: The validation/test word pairs, organized by source and target lengths.
        encoder: An encoder model to produce annotations for each step of the input sequence.
        decoder: A decoder model to generate output tokens.
        idx_dict: Contains char-to-index and index-to-char mappings, and start & end token indexes.
        criterion: Used to compute the CrossEntropyLoss for each decoder output.
        opts: The command-line arguments.

    Returns:
        mean_loss: The average loss over all batches from data_dict.
    """

    start_token = idx_dict['start_token']
    end_token = idx_dict['end_token']
    char_to_index = idx_dict['char_to_index']

    losses = []

    for key in data_dict:

        input_strings, target_strings = zip(*data_dict[key])
        # Make your input tensor and the target tensors
        # HINT : use the function string_to_index_list given in data_handler.py
        # input_tensors = ....
        # output_tensors = ....
        input_tensors, output_tensors = generate_input_tensors(char_to_index, end_token, input_strings, target_strings)
        num_tensors = len(input_tensors)
        num_batches = int(np.ceil(num_tensors / float(opts.batch_size)))

        for i in range(num_batches):

            start = i * opts.batch_size
            end = start + opts.batch_size

            # Define inputs and targets for THIS batch, beginning at index 'start' to 'end'
            # inputs = ....
            # outputs = ....
            inputs = Variable(input_tensors[start:end])
            targets = Variable(output_tensors[start:end])
            
            # The batch size may be different in each epoch
            BS = inputs.size(0)

            encoder_annotations, encoder_hidden, encoder_cell = encoder.forward(inputs)
            
            # The last hidden state of the encoder becomes the first hidden state of the decoder
            # decoder_hidden = ....
            # decoder_cell = .. either zeros, or last encoder hidden state
            decoder_hidden = Variable(encoder_hidden)
            decoder_cell = Variable(encoder_cell)
            
            # Define the first decoder input. This would essentially be the start_token
            # decoder_input = ....
            decoder_input = Variable(torch.LongTensor([start_token]*BS).unsqueeze(1))
            
            loss = 0.0

            seq_len = targets.size(1)  # Gets seq_len from BS x seq_len

            for i in range(seq_len):
                decoder_output, decoder_hidden, decoder_cell, attention_weights = decoder.forward(decoder_input,\
                                                                                    decoder_hidden,\
                                                                                    decoder_cell, \
                                                                                    encoder_annotations)

                current_target = targets[:,i]
                assert equal_shape(current_target, [BS])

                # Calculate the cross entropy between the decoder distribution and Ground truth (current_target)
                # loss += ....
                loss += criterion(decoder_output, current_target)
                
                # Find out the most probable character (ni) from the softmax distribution produced
                # ni = ....
                
                # Update decoder_input at the next time step to be this time-step's target 
                # decoder_input = ....
                decoder_input = Variable(targets[:,i].unsqueeze(1))
                assert equal_shape(decoder_input, [BS,1])

            loss /= float(seq_len)
            losses.append(loss.item())

    mean_loss = np.mean(losses)

    return mean_loss

def generate_input_tensors(char_to_index, end_token, input_strings, target_strings=False):
    '''
    Input: list of strings for input and output
    Output: matrix of corresponding tensors
    '''
    # Inputs
    input_arr, target_arr = [],[]
    # if input strings variable is a single word,
    # pass in single word to function
    if isinstance(input_strings, str):
        input_arr.append(string_to_index_list(input_strings,char_to_index, end_token))
    # else, iterate thru each word
    else: 
        for word in input_strings:
            input_arr.append(string_to_index_list(word,char_to_index, end_token))
    input_arr = torch.LongTensor(input_arr)
    
    # Outputs (optional)
    if target_strings:
        for word in target_strings:
            target_arr.append(string_to_index_list(word,char_to_index, end_token))
        target_arr = torch.LongTensor(target_arr)
    
    # Return in tensor format
    return input_arr, target_arr

# +
def find_pig_latin(sentence, encoder, decoder, idx_dict, opts):
    """Translates a sentence from English to Pig-Latin, by splitting the sentence into
    words (whitespace-separated), running the encoder-decoder model to translate each
    word independently, and then stitching the words back together with spaces between them.
    """
    return ' '.join([translate(word, encoder, decoder, idx_dict, opts) for word in sentence.split()])


def translate(input_string, encoder, decoder, idx_dict, opts):
    """Translates a given string from English to Pig-Latin.
    Not much to do here as well. Follows basically the same structure as that of the function evaluate.
    """

    char_to_index = idx_dict['char_to_index']
    index_to_char = idx_dict['index_to_char']
    start_token = idx_dict['start_token']
    end_token = idx_dict['end_token']

    max_generated_chars = 20
    gen_string = ''

    # convert given string to an array of indexes
    # HINT: use the function string_to_index_list provided in data_handler
    # indexes = ....
    indicies, _ = generate_input_tensors(char_to_index, end_token, input_string)
    
    encoder_annotations, encoder_last_hidden, encoder_cell = encoder.forward(indicies)

    # The last hidden state of the encoder becomes the first hidden state of the decoder
    # decoder_hidden = ....
    # decoder_cell = ... zeros, or last encoder hidden state
    decoder_hidden = encoder_last_hidden
    decoder_cell = encoder_cell

    # Define the first decoder input. This would essentially be the start_token
    # decoder_input = ....
    BS = 1 
    decoder_input = torch.LongTensor([start_token]*BS).unsqueeze(1)
    

    """
    CHECK ENCODER ANNOTATIONS!!!!
    """
    for i in range(max_generated_chars):
        decoder_output, decoder_hidden, decoder_cell, attention_weights = decoder.forward(decoder_input,\
                                                                            decoder_hidden,\
                                                                            decoder_cell, \
                                                                            encoder_annotations)
    
        # Calculate the cross entropy between the decoder distribution and Ground truth (current_target)
        # loss += ....

        # Find out the most probable character (ni) from the softmax distribution produced
        # ni = ....

        # soft_max_output = decoder.softmax(decoder_output)
        new_token = decoder_output.argmax()
        ni = index_to_char[new_token.item()]
        if int(new_token.item()) == end_token:
            break
        else:
            gen_string += index_to_char[new_token.item()]
            
            # update decoder_input at the next time step to be ni 
            # decoder_input = ....
            prev_prediction = torch.LongTensor([new_token]*BS).unsqueeze(1)
            decoder_input = prev_prediction 

    return gen_string


# -

try:
    train_model(train_dict, val_dict, idx_dict, encoder, decoder, opts)
except KeyboardInterrupt:
    print('Exiting early from training.')


