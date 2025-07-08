
import kagglehub
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer 
from torch.utils.data import Dataset, DataLoader
from Non_ECG_GRPO import Training, model_4bit
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb 
from sklearn.model_selection import train_test_split

# Downlad latest version
path = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
#print("Path to dataset files:", path)
#print("Files in dataset:", os.listdir(path))

#turn into dataset
data_path = os.path.join(path, "socratic_train.csv")
df = pd.read_csv(data_path)
#print(df.head())

#make final_answer column
def parse_final_answer(text):

    raw_parts = text.split("####")

    final_answer = raw_parts[-1]

    final_answer = final_answer.strip().replace(",", "")
    return float(final_answer)

df["final_answers"] = df["answer"].apply(parse_final_answer)
#print(df["final_answers"])

#make follow up questions column (list of the follow up questions)
def parse_follow_up_questions(text):
    
    text = text.split("####")[0]

    parts = []

    for p in text.split("**"):

      parts.append(p.strip())

    answers = parts[0::2]

    questions = []

    cleaned_questions = []

    for x in answers:
      if "?" in x:
        questions.append(x)


    for y in questions:
      if "\n" not in y:
        cleaned_questions.append(y)
      else:
        cleaned_questions.append(y.split("\n")[-1])
  

    return cleaned_questions
  
df["follow_up_questions"] = df["answer"].apply(parse_follow_up_questions)
#print(df["follow_up_questions"][0])

#make follow up answers column (list of the follow up answers to the follow up questions)
def parse_follow_up_answers(text):
    
    text = text.split("####")[0]

    parts = []

    for p in text.split("**"):

      parts.append(p.strip())

    answers = parts[1::2]

    clean = []

    for x in answers:
      
      text = x.split(">>")[-1]
      
      text = text.split("\n")[0]

      clean.append(text)

    return clean
  
df["follow_up_answers"] = df["answer"].apply(parse_follow_up_answers)
#print(df["follow_up_answers"])
#print(df.head())

#prompt engineering 

#question + final answer
def full_text(df):
  inputs = []

  for index, row in df.iterrows():
    #needs to generate the final answer
    prompt = "Q: " + str(row["question"]).strip() + "\n A: " + str(row["final_answers"]).strip()
    inputs.append(prompt)

  return inputs

#only the prompt
def prompt_only(df):
  inputs = []

  for index, row in df.iterrows():
    #needs to generate the final answer
    prompt = "Q: " + str(row["question"]).strip() + "\n A: " 
    inputs.append(prompt)

  return inputs

def prompt_engineering(df):
  inputs = []
    
  for index, row in df.iterrows():
     
    prompt = f"Solve this problem and provide ONLY the final numeric answer. Do not show calculation steps or follow up questions. No explanations. Q: {row['question'].strip()}\nAnswer: The answer is"
      
    inputs.append(prompt)
    
  return inputs


def extract_final_answer(generated_text_list):

    final_answer_list = []

    for text in generated_text_list:
        final_answer = text.split("A:")[-1].strip()
        final_answer_list.append(final_answer)

    return final_answer_list

#calculate prompt lengths
#not all prompts have the same length
def calculate_prompt_length(prompts, tokenizer, max_length):

  prompt_lengths = []

  for prompt_text in prompts:
        # Tokenize individual prompt without padding
        tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Get actual length (excluding padding)
        #sequence length
        actual_length = tokens["input_ids"].shape[1]
        prompt_lengths.append(actual_length)
  
  return prompt_lengths

#prepare inputs for batching
#each item in the batch should include input ids, labels, attention_mask
class TextDataset(Dataset):

    def __init__(self, full_tokenized_inputs, prompt_lengths):
        
        #only take up to the question, not the final answer of the full text
        self.input_ids = full_tokenized_inputs["input_ids"]

        #batch size, seq length
        labels = full_tokenized_inputs['input_ids'].clone()
    
        self.attention_mask = full_tokenized_inputs.get('attention_mask', None)

        #list of the prompt lengths
        self.prompt_lengths = prompt_lengths

        #mask the previous text, only final answer used in calculations
        #all rows, all columns except the last 
        #everything but the final answer should be -100
        for i in range(labels.shape[0]):
            
            labels[i, :prompt_lengths[i]] = -100

        if self.attention_mask is not None:
        # mask padding tokens, they should not be considered in loss calculation
          #padding_mask = (self.attention_mask == 0)
          #labels[padding_mask] = -100

          for i in range(self.attention_mask.shape[0]):
            for j in range(self.attention_mask.shape[1]):
              if self.attention_mask[i, j] == 0:
                labels[i, j] = -100
        
        self.labels = labels


    #batch size
    def __len__(self):
        return self.input_ids.shape[0]


   #returns item in the batch as a dictionary given it's index
    def __getitem__(self, index):
    
        item = {
        "input_ids": self.input_ids[index],
        "labels": self.labels[index]           }
        
        if self.attention_mask is not None:
        
            item["attention_mask"] = self.attention_mask[index]
        
        return item

class RLDataset(Dataset):

  def __init__(self, encoded_prompts, final_answers):

    self.encoded_prompts = encoded_prompts
    self.final_answers = final_answers
    self.input_ids = self.encoded_prompts["input_ids"]
    self.attention_mask = self.encoded_prompts["attention_mask"]


  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, index):

    item = {

      "input_ids": self.input_ids[index],
      "attention_mask": self.attention_mask[index],
      "true_answer": self.final_answers[index]
    }

    return item



#rewards function gives the output a score of 1.0 if the final answer matches the actual final answer else 0.0 score
def rewards_functionX(generated_answer_list, final_answer_list):
    
    list_of_scores = []

    for i, text in enumerate(generated_answer_list):
        outputted_answer = None
        answer = final_answer_list[i]

        for word in text.split():
            if word.isdigit():
                outputted_answer = word
                break

        if outputted_answer is not None and str(outputted_answer) == str(answer):
            list_of_scores.append(1.0)
        
        else:
            list_of_scores.append(0.0)

    return list_of_scores

def rewards_function(generated_answer_list, final_answer_list):
    
    list_of_scores = []

    for i, text in enumerate(generated_answer_list):

      answer = final_answer_list[i]
      if answer == generated_answer_list[i]:
        list_of_scores.append(1.0)
        
      else:
        list_of_scores.append(0.0)

    return list_of_scores


df["tight_prompt"] = prompt_engineering(df)

df_sft, df_rl = train_test_split(df, test_size=0.2, random_state=42)

df_sft = df_sft.sample(frac=0.1, random_state=42).reset_index(drop=True)

df_rl = df_rl.sample(frac=0.1, random_state=42).reset_index(drop=True)

full_text_sft = full_text(df_sft)
#print(full_text)

prompt_sft = prompt_only(df_sft)
#print(prompt)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.3"

#loading the model
model = AutoModelForCausalLM.from_pretrained(model_name)

#loading the tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name)

#make the padding token the end of sequence token
tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 512

#tokenizing the full text
#max length so all of the sequences are the same length (pad to the max length, truncate to max length)
tokenized_text_sft = tokenizer(
    full_text_sft,
   padding=True,
    truncation=True,
    max_length = MAX_LENGTH,
    return_tensors="pt")

tokenized_prompt_sft = tokenizer(
    prompt_sft,
   padding=True,
    truncation=True,
    max_length = MAX_LENGTH,
    return_tensors="pt")

#print(tokenized_text["input_ids"])
#decoded_text = tokenizer.decode(tokenized_text["input_ids"][0], skip_special_tokens=True)
#print(decoded_text)

#full_ids = tokenized_text["input_ids"][0]
#print(len(full_ids))

tokenized_prompt_rl = tokenizer(
    df_rl["tight_prompt"].tolist(),
   padding=True,
    truncation=True,
    max_length = MAX_LENGTH,
    return_tensors="pt")

prompt_lengths_sft = calculate_prompt_length(prompt_sft, tokenizer, MAX_LENGTH)

#turn the tokenized inputs into batches
dataset_sft = TextDataset(tokenized_text_sft, prompt_lengths_sft)

dataset_rl = RLDataset(tokenized_prompt_rl, df_rl["final_answers"].tolist())



length = dataset_sft.__len__()
#print(length)

#tenth = dataset_sft.__getitem__(10)
#print(tenth)

#define how many batches to break up data into

batch_size = 8
#make into batches
dataloader_sft = DataLoader(dataset_sft, batch_size=batch_size, shuffle=True)

dataloader_rl = DataLoader(dataset_rl, batch_size=batch_size, shuffle=True)

#dataset_grpo = RLDataset(tokenized_tight_prompt_rl, final_answers_rl)

#print(dataloader.__len__())

#make model 4 bit
quant_model = model_4bit(model)
#for name, module in quant_model.named_parameters():
#  print(name)

#r and lora_alpha same value

lora_config = {

  "r": 16, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.1, "bias": "none", "task_type":TaskType.CAUSAL_LM}

#optim already default defined in training class 
#learning rate default defined in training class
#device default defined in training class
#text tokenizer default defined in training class
#weight decay default defined in training class

pipeline_model = Training(reward_model = rewards_function, model = quant_model, lora_config = lora_config)

#for name, param in pipeline_model.model.named_parameters():
#    if "lora" in name:
#        print(name, param.data.mean().item())

#
#pipeline_model.train_adapters(dataloader_sft, num_epochs=2, gradient_accumulation_steps=8)

#making the path (does not exist yet)
adapter_path = "/home/sirfan/ECG_tokenizer/models/adapter_weights"

#save the weights at the path
#pipeline_model.save_adapter_checkpoint(adapter_path)

#generate reports
#reports = pipeline_model.generate_initial_reports(dataloader_sft, adapter_path, max_new_tokens=128)
#print(reports)

#train the model


pipeline_model.grpo(dataloader_rl, adapter_path, epochs=4, max_new_tokens = 128, num_candidates=2)




