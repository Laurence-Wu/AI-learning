from context import read_files_from_folder
from learn_bpe import get_vocabulary, learn_bpe, open_file
import os
with read_files_from_folder('bpe_code') as files:
    with open("current_vocab.txt",'w') as outputfiles:
        learn_bpe(files,outputfiles,1000,total_symbols=True)
    

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def tokenize_text(text, output_file):
    tok_name = "bert-base-uncased"        # <— swap to "gpt2", "facebook/opt-1.3b", etc.
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

    def collate(batch):
        # `batch` is a list of raw strings
        enc = tokenizer(batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    dataloader = DataLoader(raw_text_dataset,
                            batch_size=32,
                            collate_fn=collate)     # <-- model ready
