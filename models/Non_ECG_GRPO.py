from peft import LoraConfig, get_peft_model, PeftModel
import bitsandbytes as bnb 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW
import os 
import torch
from typing import Optional
import re

def model_4bit(model):
    for name, module in model.named_modules():
        
        if isinstance(module, nn.Linear):
            # Create a new Linear4bit layer with the same config as the original
            quantized_layer = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=(module.bias is not None),
            quant_type='nf4',
            compute_dtype=torch.float16
)

            # Copy weights and bias from the original linear layer
            quantized_layer.weight.data.copy_(module.weight.data)
            
            if module.bias is not None:
                
                quantized_layer.bias.data.copy_(module.bias.data)
            
            # Now replace the original linear layer in the model with quantized one
            parent = model
            
            parts = name.split(".")
            
            for part in parts[:-1]:
                
                parent = getattr(parent, part)
            
            setattr(parent, parts[-1], quantized_layer)
    
    return model

def expand_final_answers(final_answers_list, num_candidates):

    expanded_final_answers = []

    for x in final_answers_list:

        for y in range(num_candidates):

            expanded_final_answers.append(x)
    
    return expanded_final_answers


#from the outputted generated text extract the final answer
#needed to compare with the actual final answer in the rewards function
import re

def extract_final_answer(text_list):
    final_answers = []
    for block in text_list:
        # Find the first occurrence of A: number (e.g., A: 36 or A: 36s)
        match = re.search(r"A:\s*([$€]?[0-9]+(?:\.[0-9]+)?)(s)?", block)
        if match:
            final_answers.append(match.group(1))
        else:
            # fallback: first number in the block
            fallback = re.search(r"[$€]?[0-9]+(?:\.[0-9]+)?", block)
            final_answers.append(fallback.group(0) if fallback else "NO_ANSWER")
    return final_answers


class Training:

    def __init__(self, reward_model, 
        model = None, 
        lora_config: Optional[dict] = None, 
        optimizer = None, 
        learning_rate = None, 
        device = None, 
        text_tokenizer = None,
        weight_decay = None):

        self.reward_model = reward_model

        if model == None:
            model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3")
        
        model = model_4bit(model)
        self.model = model

        #freeze base weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        #lora_config is a dictionary
        if lora_config is not None:
            self.model = get_peft_model(self.model, LoraConfig(**lora_config))
        
        if learning_rate == None:            
            learning_rate = 1e-5
       
        self.learning_rate = learning_rate

        if weight_decay == None:
            weight_decay = 0.1
        
        self.weight_decay = weight_decay

        if optimizer == None:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        self.optimizer = optimizer

        if device == None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        self.device = device

        #to decode the text when using .generate()
        if text_tokenizer == None:
            text_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3")
        
        self.text_tokenizer = text_tokenizer
      

    
    def train_adapters(self, dataloader, num_epochs = 2, gradient_accumulation_steps = 8):

        #Example batch
        #Batch 
        #{"input_ids": torch.LongTensor of shape (batch_size, seq_len),
        #"labels": torch.LongTensor of shape (batch_size, seq_len),
        #"attention_mask": (optional) torch.LongTensor of shape (batch_size, seq_len)}

        self.model.train()
        self.model.to(self.device)

        #cost function
        #expects number of samples, number of classes (vocab size)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(num_epochs):
            
            total_loss = 0.0

            for step, batch in enumerate(dataloader):
                #token IDs that represent the input text
                input_ids = batch["input_ids"].to(self.device)

                labels = batch["labels"].to(self.device)

                attention_mask = batch.get('attention_mask', None)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
               
                #forward pass
                outputs = self.model(input_ids = input_ids,
                                                labels = labels,
                                                attention_mask = attention_mask,
                                                return_dict = True)

                #raw scores
                #batch size, sequence length, vocab size
                logits = outputs.logits

                #input: [N, C] scores for C classes for N samples,
                #target: true classes 
               
                reshaped = logits.view(-1, logits.shape[-1]) #batch * seq_len, vocab_size
                loss = criterion(reshaped, 
                                labels.view(-1)) #list of labels 1D batch *seq_len
                
                #normalize loss
                loss = loss/gradient_accumulation_steps 

                #back propogation
                #accumulating the gradients


                #print(f"Step {step} logits stats: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
                #print(f"Step {step} labels valid count: {(labels != -100).sum().item()}")
                #print(f"Step {step} loss: {loss.item()}")


                loss.backward()
                
                #returns tensor number
                total_loss += loss.item()

                #if reached the gradient_accumulation_step 
                #force optimization step at the last batch
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    #optimizing step
                  
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                  
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            avg_loss = total_loss /len(dataloader)

            print(f"Training adapters: \n Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
    def save_adapter_checkpoint(self, path):

        #make the directory if it doesn't exist 
        os.makedirs(path, exist_ok = True)

        #saves the model's state as files in a directory
        self.model.save_pretrained(path)
        print(f"Adapter saved at path {path}")


    def generate_initial_reports(self, dataloader, adapter_path, max_new_tokens = 128):

        #input saved adapters in the file into base model
        self.model.load_adapter(adapter_path, adapter_name="lora_adapter")

        self.model.to(self.device)

        self.model.eval()

        #the list containing all the outputs generated 
        all_reports = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)

            attention_mask = batch.get("attention_mask", None)
            
            if attention_mask is not None:

              attention_mask = attention_mask.to(self.device)
            
            #turn off gradients for faster inference
            with torch.no_grad():

                #beams are deterministic 
                #generates token ID sequences
                generated_ids = self.model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = max_new_tokens, #number of tokens that can be generated after input seq length
                    do_sample = False,
                    num_beams = 4,
                    num_return_sequences = 1,
                    temperature=0
                )
               
            #converts each sequence of tokens into string
            #a list
            reports = self.text_tokenizer.batch_decode(generated_ids, skip_special_tokens = True)

            print(reports)

            #adding the text generated from this batch 
            all_reports.extend(reports)

        return all_reports


    def grpo(self, final_answer_list, dataloader, adapter_path, epochs = 4, max_new_tokens = 10, num_candidates = 2):

        self.model.load_adapter(adapter_path, adapter_name="lora_adapter")

        self.model.to(self.device)

        self.model.train()

        for epoch in range(epochs):

            batch_number = 0

            for batch in dataloader:

                outputs_for_batch = []

                log_probs_list = []

                input_ids = batch["input_ids"].to(self.device)

                attention_mask = batch.get("attention_mask", None)
            
                if attention_mask is not None:

                  attention_mask = attention_mask.to(self.device)

                #using do sample for more variability 
                generated_ids = self.model.generate(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        max_new_tokens = max_new_tokens,
                        do_sample = False,
                        num_beams=4,
                        num_return_sequences = num_candidates #generate num candidates output sequence per input 
                    ) #batch_size * num_return_sequences, sequence_length + max_new_tokens
                
                #decode the outputs for the batch
                #list of stringss
                outputs_for_batch = self.text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                print(outputs_for_batch[0])

                input_batch_size = input_ids.shape[0] #batch size, seq length
                #print(input_batch_size)

                #generated final answers
                final_answers = extract_final_answer(outputs_for_batch)
                #print(final_answers)
                #print(len(final_answers))

                expanded_final_answers_list = expand_final_answers(final_answer_list, num_candidates)
                #print(expanded_final_answers_list)

                #return list of scores
                scores = self.reward_model(final_answers, final_answer_list)
                #print(scores)

                scores_tensor = torch.tensor(scores, device = self.device)
                #print(scores_tensor)

                #each row in each batch is an input, each column is output generated for that input
                scores_tensor = scores_tensor.view(input_batch_size, num_candidates)

                #shape batch size, 1 
                #dim = 1 takes the average across second dim num_candidates
                avg_scores = scores_tensor.mean(dim=1, keepdim=True)

                #batch size, num_candidates
                advantage_tensor = scores_tensor - avg_scores

                for candidate in outputs_for_batch:

                    #tokenzie the output 
                    tokens = self.text_tokenizer(candidate, return_tensors = "pt").to(self.device)

                    input_ids_c = tokens["input_ids"]
                    attention_mask_c = tokens["attention_mask"]

                    output = self.model(input_ids = input_ids_c,
                                                attention_mask = attention_mask_c
                                                #need the ecg signals for the specific batch that the candidate was produced from
                                                #unsqueeze to make it back to (1, channels, seq length)
                                                )
                        
                    logits = output.logits #shape [1, seq len, vocab_size]

                    #apply softmax on last dimension
                    #log probabilities for all vocab tokens at every position vocab_size
                    log_probs = F.log_softmax(logits, dim=-1)

                    #drop the prediction at the last token, since last token nothing to predict afterwards
                    shifted_log_probs = log_probs[:, :-1, :]

                    #drop the BOS token
                    shifted_input_ids = input_ids_c[:,1:] #batch, seq length

                    #add third dimension to match shifted_log_probs
                    shifted_input_ids = shifted_input_ids.unsqueeze(-1)

                    #Extracts the log-probability of the correct token at each position
                    token_log_probs = shifted_log_probs.gather(2, shifted_input_ids) #[batch_size, seq_len, 1]

                    #remove the last dimension
                    #batch size = 1
                    token_log_probs = token_log_probs.squeeze(-1) #[batch_size, seq_len]

                    #takes the sum of the token logs since batch size is 1
                    log_prob_sum = token_log_probs.sum()

                    #add to log probs list
                    log_probs_list.append(log_prob_sum)

                log_probs_tensor = torch.stack(log_probs_list)
            
                log_probs_tensor = log_probs_tensor.view(input_batch_size, num_candidates)

                #compute GRPO loss
                loss = -(log_probs_tensor * advantage_tensor).mean()

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                batch_number += 1 

                print(f"GRPO: \n Epoch {epoch} | Batch number {batch_number} | Loss : {loss.item():.4f}")



