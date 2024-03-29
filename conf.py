"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

import torch
import sentencepiece as spm

bpe_flag= True
flag_distill= True

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers=8
shuffle=True

# model parameter setting
batch_size = 64
n_batch_split = 4
max_len = 2000
d_model = 256
n_encoder_layers=2
n_decoder_layers=6
n_heads = 8
n_enc_replay = 6
dim_feed_forward= 2048
drop_prob = 0.1
depthwise_kernel_size=31
max_utterance_length= 600 #max nummber of labels in training utterances

src_pad_idx=0
trg_pad_idx=30
trg_sos_idx=1
trg_eos_idx=31
enc_voc_size=32
dec_voc_size=32

lexicon="lexicon.txt"
tokens="tokens.txt"

sp = spm.SentencePieceProcessor()
if bpe_flag == True:
    sp.load('sentencepiece/build/libri.bpe-256.model')
    src_pad_idx=0
    trg_pad_idx=126
    trg_sos_idx=1
    trg_eos_idx=2
    enc_voc_size=sp.get_piece_size()
    dec_voc_size=sp.get_piece_size()
    lexicon="sentencepiece/build/librispeech-bpe-256.lex"
    tokens="sentencepiece/build/librispeech-bpe-256.tok"

sample_rate = 16000
n_fft = 512
win_length = 320 #20ms
hop_length = 160 #10ms
n_mels = 80
n_mfcc = 80


# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 1e-9#5e-9
patience = 10
warmup = 8000 #dataloader.size()
epoch = 10000
clip = 1.0
weight_decay = 5e-4
#weight_decay = 0.1 # pytorch transformer class 
inf = float('inf')
