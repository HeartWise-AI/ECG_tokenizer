import kagglehub
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from COT_GRPO import Training, model_4bit
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb 
from sklearn.model_selection import train_test_split
import random 
import re

# Downlad latest version
path = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
#print("Path to dataset files:", path)
#print("Files in dataset:", os.listdir(path))

#turn into dataset
data_path = os.path.join(path, "socratic_train.csv")
df = pd.read_csv(data_path)

def rewards_function(generated_answer_list, final_answer_list):
    
    list_of_scores = []

    for i, text in enumerate(generated_answer_list):

      answer = final_answer_list[i]
      if answer == generated_answer_list[i]:
        list_of_scores.append(1.0)
        
      else:
        list_of_scores.append(0.0)

    return list_of_scores

def parse_final_answer(text):

    raw_parts = text.split("####")

    final_answer = raw_parts[-1]

    final_answer = final_answer.strip().replace(",", "")
    return float(final_answer)

df["final_answers"] = df["answer"].apply(parse_final_answer)
#print(df["final_answers"])

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

def is_number(number):
    
  pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'
  number = str(number).strip().replace(",", ".")

  return bool(re.match(pattern, number))


#make follow up answers column (list of the follow up answers to the follow up questions)
def parse_numeric_follow_up_answers(text):

    end_symbols = ["$", "L", "%", "m^2", "€"]

    end_words = ["during", "dollars", "/week", "mph"]

    text = text.split("####")[0]

    parts = []

    for p in text.split("**"):

      parts.append(p.strip())

    answers = parts[1::2]

    clean = []

    for x in answers:

      text = x.split(">>")[-1]

      text = text.split("\n")[0]

      #print(text)

      if "=" in text:

        text = text.split("=")[-1]
      
      for y in end_symbols:

        if y in text:

          text = text.split(y)[0]
      
      for z in end_words:

        if z in text:

          text = text.split(z)[0]
      
      for word in text.split():
        
        if is_number(word):

          text = word

          break
        
        else:

          text = "0.0"
      
      text = text.replace(",", ".")

      if text == " " or text == "":

        text = "0.0"

      
      text.strip()

      text = float(text)

      #print(text)

      clean.append(text)

    return clean

df["numeric_follow_up_answers"] = df["answer"].apply(parse_numeric_follow_up_answers)


def thinking(df, thinking_list):

  words = []

  for x in range(len(df)):
    
    choice = random.choice(thinking_list)

    words.append(choice)
  
  return words

thinking_list = ["let's solve it step by step", "let's think it through", "Thinking about the question", "the various steps to solve this question"]


df["thinking"] = thinking(df, thinking_list)


#question + follow up questions + follow up answers + final answer
def full_text(df):
  inputs = []

  for index, row in df.iterrows():

    thoughts = []

    length = min(len(row["follow_up_questions"]), len(row["follow_up_answers"]))

    for x in range(length):

      chain_of_thought = "Q" + str(x) + ": " + str(row["follow_up_questions"][x]) + " A" + str(x) + ": " + str(row["follow_up_answers"][x])

      thoughts.append(chain_of_thought)

    thought = " ".join(thoughts)

    prompt = f"Question: " + str(row["question"]).strip() + " \n T: " + str(row["thinking"]).strip() + " " + thought + "\n Final Answer: " + str(row["final_answers"]).strip()

    inputs.append(prompt)

  return inputs


def prompt_only(df):
    inputs = []

    for index, row in df.iterrows():
        #needs to generate the final answer
        
        thoughts = []

        length = len(row["follow_up_questions"])

        chain_of_thought = f"Provide a concise but complete answer for each step. If the answer requires calculation, show it briefly. Question: " + str(row["question"]).strip() + " \n T: " + str(row["thinking"]).strip() + " "
    
        for x in range(length):

            thought = "\n Q" + str(x) + ": " + str(row["follow_up_questions"][x]) + " A" + str(x) + ": "

            chain_of_thought = chain_of_thought + thought

            thoughts.append(chain_of_thought)

        prompt = f"\n Final Answer: "

        chain_of_thought = chain_of_thought + prompt

        thoughts.append(chain_of_thought)

        inputs.append(thoughts)
    
    return inputs

df["tight_prompt"] = prompt_only(df) 

def get_step_answers(row):
    answers = []
    q_len = len(row["follow_up_questions"])
    a_len = len(row["numeric_follow_up_answers"])
    steps = min(q_len, a_len)
    for i in range(steps):
        answers.append(row["numeric_follow_up_answers"][i])

    if len(answers) == 0 or answers[-1] != row["final_answers"]:
        answers.append(row["final_answers"])


    return answers

df["step_answer"] = df.apply(get_step_answers, axis=1)

df = df.groupby("question", as_index=False).agg({
    "tight_prompt": "first",
    "step_answer": "first",
    "answer": "first",
    "follow_up_questions": "first",
    "follow_up_answers": "first",
    "final_answers": "first",
    "thinking": "first"
})

def pair_together(df):

  pairs = []

  for index, row in df.iterrows():

    minimum = min(len(row["step_answer"]), len(row["tight_prompt"]))

    row_pairs = []

    for x in range(minimum):

      pair = (row["tight_prompt"][x], row["step_answer"][x])

      #print(pair, type(pair))

      row_pairs.append(pair)
    
    pairs.append(row_pairs)
  
  return pairs

df["prompt_answer_pair"] = pair_together(df)

df = df.explode("prompt_answer_pair").reset_index(drop=True)

df[["prompt", "step_answer"]] = pd.DataFrame(df["prompt_answer_pair"].tolist(), index=df.index)
#print(df["step_answer"].tolist()) 

#prepare inputs for batching
#each item in the batch should include input ids, labels, attention_mask
class TextDataset(Dataset):

    def __init__(self, full_tokenized_inputs, labels, attention_mask):
        
        #only take up to the question, not the final answer of the full text
        self.input_ids = full_tokenized_inputs

        self.attention_mask = attention_mask

        if self.attention_mask is not None:
        # mask padding tokens, they should not be considered in loss calculation
          #padding_mask = (self.attention_mask == 0)
          #labels[padding_mask] = -100

          for i in range(len(self.attention_mask)):
            for j in range(len(self.attention_mask[i])):
              if self.attention_mask[i][j] == 0:
                labels[i][j] = -100
        
        self.labels = labels


    #batch size
    def __len__(self):
        return len(self.input_ids)


   #returns item in the batch as a dictionary given it's index
    def __getitem__(self, index):
    
        item = {
        "input_ids": self.input_ids[index],
        "labels": self.labels[index]           }
        
        if self.attention_mask is not None:
        
            item["attention_mask"] = self.attention_mask[index]
        
        return item

class RLDataset(Dataset):

  def __init__(self, encoded_prompts, step_answers):

    self.encoded_prompts = encoded_prompts
    self.step_answers = step_answers
    self.input_ids = self.encoded_prompts["input_ids"]
    self.attention_mask = self.encoded_prompts["attention_mask"]

  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, index):

    item = {

      "input_ids": self.input_ids[index],
      "attention_mask": self.attention_mask[index],
      "step_answers": self.step_answers[index]
    }

    return item

df_sft, df_rl = train_test_split(df, test_size=0.2, random_state=42)

df_sft = df_sft.sample(frac=0.1, random_state=42).reset_index(drop=True)

df_rl = df_rl.sample(frac=0.1, random_state=42).reset_index(drop=True)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.3"

#loading the model
model = AutoModelForCausalLM.from_pretrained(model_name)

#loading the tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name)

#make the padding token the end of sequence token
tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 512

#manually creates labels and premasks tokens that are won't be used in calculations
def create_labels(df, tokenizer):
  all_labels = []
  all_inputs = []
  all_attention_masks = []

  for index, row in df.iterrows():

    labels = []
    inputs = ""

    main_question = "Question: " + str(row["question"]).strip() + " \n T: " + str(row["thinking"]).strip() + " "

    main_question_tokens = tokenizer(
            main_question,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
    
    labels += [-100] * main_question_tokens["input_ids"].size(1)
    inputs += main_question

    length = min(len(row["follow_up_questions"]), len(row["follow_up_answers"]))
    
    for x in range(length):

      question = "Q" + str(x) + ": " + str(row["follow_up_questions"][x]) + " A" + str(x) + ": " 

      inputs+= question

      answer = str(row["follow_up_answers"][x])

      inputs+=answer

      question_tokens = tokenizer(
            question,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
      
      #input ids 1, seq length
      labels += [-100] * question_tokens["input_ids"].size(1)
      
      answer_tokens = tokenizer(
            answer,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

      labels += answer_tokens["input_ids"][0].tolist()
    
    final_words = "\n Final Answer: " 

    inputs+=final_words

    final_words_tokens = tokenizer(
            final_words,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
    
    labels += [-100]*final_words_tokens["input_ids"].size(1)

    final_answer = str(row["final_answers"]).strip()

    inputs+=final_answer 

    final_answer_tokens = tokenizer(
            final_answer,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

    labels += final_answer_tokens["input_ids"][0].tolist()

    full_text_tokens = tokenizer(
            inputs,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
    
    input_ids = full_text_tokens['input_ids'][0]

    attention_mask = full_text_tokens["attention_mask"][0]

    #making labels same length as input_ids
    labels_tensor = torch.tensor(labels[:len(input_ids)], dtype=torch.long)

    if len(labels_tensor) < len(input_ids):
      
      pad_len = len(input_ids) - len(labels_tensor)
      
      labels_tensor = torch.cat([labels_tensor, torch.full((pad_len,), -100)])
    
    elif len(labels_tensor) > len(input_ids):
    
      labels = labels[:len(input_ids)]
    
    all_inputs.append(full_text_tokens)

    all_labels.append(labels_tensor)

    all_attention_masks.append(attention_mask)

  return all_labels, all_inputs, all_attention_masks



labels, inputs, attention_mask = create_labels(df_sft, tokenizer)
#print(len(labels))
#print(len(inputs))
#print(attention_mask)

#print(type(inputs[0]))
#print(type(labels[0]))
#print(type(attention_mask[0]))

#turns into tensor
inputs = [x["input_ids"].squeeze(0) for x in inputs]

padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)


#tokenizing the full text
#max length so all of the sequences are the same length (pad to the max length, truncate to max length)

#chains = prompt_only(df_rl)
#flattened_prompts = [step for row in chains for step in row]

prompts = df_rl["prompt"].tolist()

tokenized_prompt_rl = tokenizer(
    prompts,
   padding=True,
    truncation=True,
    max_length = MAX_LENGTH,
    return_tensors="pt")

answers = df_rl["step_answer"].tolist()


#turn the tokenized inputs into batches
dataset_sft = TextDataset(padded_inputs, padded_labels, padded_attention_mask)

dataset_rl = RLDataset(tokenized_prompt_rl, answers)

length = dataset_sft.__len__()
#print(length)

#tenth = dataset_sft.__getitem__(10)
#print(tenth)

batch_size = 8
#make into batches
dataloader_sft = DataLoader(dataset_sft, batch_size=batch_size, shuffle=True)

for batch in dataloader_sft:
    print("Batch input_ids:", batch["input_ids"].shape)
    print("Batch labels:", batch["labels"].shape)
    print("Batch mask:", batch["attention_mask"].shape)
    break

#print("dataset_sft length:", len(dataset_sft))
#print("First item:", dataset_sft[0])

dataloader_rl = DataLoader(dataset_rl, batch_size=batch_size, shuffle=True)


#make model 4 bit
#quant_model = model_4bit(model)

#r and lora_alpha same value
lora_config = {

  "r": 16, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.1, "bias": "none", "task_type":TaskType.CAUSAL_LM}

#optim already default defined in training class 
#learning rate default defined in training class
#device default defined in training class
#text tokenizer default defined in training class
#weight decay default defined in training class

pipeline_model = Training(reward_model = rewards_function, model = model, lora_config = lora_config)

#print(df.columns.tolist())
#print(df_rl["prompt"][0])
#print(df_rl["step_answer"][0])
#print(df_rl["final_answers"][0])

#train adapters
#pipeline_model.train_adapters(dataloader_sft, num_epochs=2, gradient_accumulation_steps=8)

#making the path (does not exist yet)
adapter_path = "/home/sirfan/ECG_tokenizer/models/cot_adapter_weights"

#save the weights at the path
#pipeline_model.save_adapter_checkpoint(adapter_path)

#generate reports
#reports = pipeline_model.generate_initial_reports(dataloader_rl, adapter_path, max_new_tokens=10)
#print(reports)

#train the model
pipeline_model.grpo(dataloader_rl, adapter_path, epochs=4, max_new_tokens = 128, num_candidates=1)


