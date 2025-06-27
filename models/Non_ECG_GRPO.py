from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW
import os 
import torch
from typing import Optional

def model_4bit(model):

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):

            quantized_layer = bnb.nn.Linear4bit.from_linear(
              module,
              quant_type='nf4',
              compute_dtype=torch.float16
)            
            parent = model

            parts = name.split(".")

            for part in parts[:-1]:

                #same as parent.part
                parent = getattr(parent, part)

            #parent now the container for the layer to replace 
            setattr(parent, parts[-1], quantized_layer)
    
    return model 

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
            learning_rate = 2e-4
       
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
        criterion = nn.CrossEntropyLoss()

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
                outputs = self.model.forward_ecg(input_ids = input_ids,
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
                loss.backward()
                
                #returns tensor number
                total_loss += loss.item()

                #if reached the gradient_accumulation_step 
                #force optimization step at the last batch
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    #optimizing step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            avg_loss = total_loss /len(dataloader)

            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
    def save_adapter_checkpoint(self, path):

        #make the directory if it doesn't exist 
        os.makedirs(path, exist_ok = True)

        #saves the model's state as files in a directory
        self.model.save_pretrained(path)
        print(f"Adapter saved at path {path}")


    def generate_initial_reports(self, dataloader, base_model, adapter_path, max_new_tokens = 128):

        #input saved adapters in the file into base model
        adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

        adapter_model.to(self.device)

        adapter_model.eval()

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
                generated_ids = adapter_model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = max_new_tokens, #number of tokens that can be generated after input seq length
                    do_sample = False,
                    num_beams = 4,
                    num_return_sequences = 1
                )

            #converts each sequence of tokens into string
            #a list
            reports = self.text_tokenizer.batch_decode(generated_ids, skip_special_tokens = True)

            #adding the text generated from this batch 
            all_reports.extend(reports)

        return all_reports


    def grpo(self, dataloader, adapter_path, epochs = 4, max_new_tokens = 128, num_candidates = 4):

        adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

        adapter_model.to(self.device)

        adapter_model.train()

        for epoch in range(epochs):

            batch_number = 0

            for batch in dataloader:

                outputs_for_batch = []

                log_probs_list = []

                input_ids = batch["input_ids"].to(self.device)

                attention_mask = batch.get("attention_mask", None)
            
                if attention_mask is not None:

                  attention_mask = attention_mask.to(self.device)

                for x in range(num_candidates):

                    #using do sample for more variability 
                    generated_ids = adapter_model.generate(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        max_new_tokens = max_new_tokens,
                        do_sample = True,
                        top_k = 50, #only sample from the top 50 most likely
                        top_p = 0.95, #sample from smallest set of possible tokens whose prob above 95%
                        num_return_sequences = 1 #generate 1 output sequence per input 
                    ) #batch_size * num_return_sequences, sequence_length + max_new_tokens

                    #a list
                    candidates = self.text_tokenizer.batch_decode(generated_ids, skip_special_tokens = True)

                    outputs_for_batch.extend(candidates)

                input_batch_size = input_ids.shape[0] #batch size, seq length

                final_scores = extract_final_answer(outputs_for_batch)

                #return list of scores
                scores = self.reward_model(outputs_for_batch, final_scores)

                scores_tensor = torch.tensor(scores, device = self.device)

                #each row in each batch is an input, each column is output generated for that input
                scores_tensor = scores_tensor.view(input_batch_size, num_candidates)

                avg_score_list = []

                for x in range(input_batch_size):

                    #Average score across all candidates for each input in the batch 
                    avg_score_list.append(scores_tensor[x].mean())
                
                #batch size, num_candidates
                advantage_tensor = torch.empty_like(scores_tensor)

                for i in range(input_batch_size):

                    for j in range(num_candidates):

                        advantage_tensor[i][j] = scores_tensor[i][j] - avg_score_list[i]

                for i, candidate in enumerate(outputs_for_batch):

                    #tokenzie the output 
                    tokens = self.text_tokenizer(candidate, return_tensors = "pt").to(self.device)

                    input_ids_c = tokens["input_ids"]
                    attention_mask_c = tokens["attention_mask"]

                    with torch.no_grad():

                        output = adapter_model(input_ids = input_ids_c,
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

                        log_prob_sum = 0.0

                        token_log_seq_length = token_log_probs.shape[-1]

                        #for every output in the output batch list
                        for j in range(token_log_seq_length):

                            log_prob_sum += token_log_probs[0][j].item()
                        
                        log_probs_list.append(log_prob_sum)

            log_probs_tensor = torch.tensor(log_probs_list, device = self.device)
            
            log_probs_tensor = log_probs_tensor.view(input_batch_size, num_candidates)

            #compute GRPO loss
            loss = -(log_probs_tensor * advantage_tensor).mean()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            batch_number += 1 

            print(f"Epoch {epoch} | Batch number {batch_number} | Loss : {loss.item():.4f}")



