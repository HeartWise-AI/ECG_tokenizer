
import kagglehub
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer 
from torch.utils.data import Dataset, DataLoader
from Non_ECG_GRPO import Training, model_4bit
from peft import LoraConfig, get_peft_model, TaskType

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

    return final_answer

df["final_answers"] = df["answer"].apply(parse_final_answer)
#print(df["final_answers"])

#make follow up questions column (list of the follow up questions)
def parse_follow_up_questions(text):
    
    text = text.split("####")[0]

    parts = []

    for p in text.split("**"):

      parts.append(p.strip())

    questions = parts[0::2]

    return questions
  
df["follow_up_questions"] = df["answer"].apply(parse_follow_up_questions)
#print(df["follow_up_questions"])

#make follow up answers column (list of the follow up answers to the follow up questions)
def parse_follow_up_answers(text):
    
    text = text.split("####")[0]

    parts = []

    for p in text.split("**"):

      parts.append(p.strip())

    answers = parts[1::2]

    return answers
  
df["follow_up_answers"] = df["answer"].apply(parse_follow_up_answers)
#print(df["follow_up_answers"])
#print(df.head())

#prompt engineering 
def creating_prompts(df):
  inputs = []

  for index, row in df.iterrows():
    prompt = "Q: " + str(row["question"]).strip() + "\nA:" + str(row["final_answers"]).strip()
    inputs.append(prompt)

  return inputs

#prepare inputs for batching
#each item in the batch should include input ids, labels, attention_mask
class TextDataset(Dataset):

    def __init__(self, tokenized_inputs):
        self.input_ids = tokenized_inputs['input_ids']
        self.labels = tokenized_inputs['input_ids'].clone()
        
        self.attention_mask = tokenized_inputs.get('attention_mask', None)

    #for Cross entropy padded tokens need labels of -100 to not be included in the calculations
        if self.attention_mask is not None:
            for i in range(self.attention_mask.shape[0]):       # batch dimension
                for j in range(self.attention_mask.shape[1]):   # sequence length dimension
                    if self.attention_mask[i, j] == 0:
                        self.labels[i, j] = -100

    #batch size
    def __len__(self):
        return self.input_ids.shape[0]


    #returns item in the batch as a dictionary given it's index
    def __getitem__(self, index):
    
        item = {
        "input_ids": self.input_ids[index],
        "labels": self.labels[index]
            }
        
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[index]
        
        return item

#from the outputted generated text extract the final answer
#needed to compare with the actual final answer in the rewards function
def extract_final_answer(generated_text_list):

    final_answer_list = []

    for text in generated_text_list:
        final_answer = text.split("A:")[-1].strip()
        final_answer_list.append(final_answer)

    return final_answer_list


#rewards function gives the output a score of 1.0 if the final answer matches the actual final answer else 0.0 score
def rewards_function(generated_text_list, final_answer_list):
    
    list_of_scores = []

    for i, text in enumerate(generated_text_list):
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

prompts = creating_prompts(df)
#print(prompts)

final_list = extract_final_answer(prompts)
#print(final_list)

#loading the model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3")

#loading the tokenizer 
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3")

#make the padding token the end of sequence token
tokenizer.pad_token = tokenizer.eos_token

#tokenizing the inputs
tokenized_inputs = tokenizer(
    prompts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

#turn the tokenized inputs into batches
dataset = TextDataset(tokenized_inputs)
#length = dataset.__len__()
#print(length)

#tenth = dataset.__getitem__(10)
#print(tenth)

#define how many batches to break up data into
batch_size = 8
#make into batches
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#print(dataloader.__len__())

#make model 4 bit
quant_model = model_4bit(model)
for name, module in quant_model.named_parameters():
    print(name)

r = 16

lora_config = LoraConfig(
    r = r,
    lora_alpha = r,
    target_modules = ["q_proj",  "v_proj"],
    lora_dropout = 0.1,
    bias = "none",
    task_type=TaskType.CAUSAL_LM
)

#optim already default defined in training class 
#learning rate default defined in training class
#device default defined in training class
#text tokenizer default defined in training class
#weight decay default defined in training class

pipeline_model = Training(reward_model = rewards_function, model = quant_model, lora_config = lora_config)

gipeline_model.train_adapters(dataloader, num_epochs=2, gradient_accumulation_steps=8)

#making the path (does not exist yet)
adapter_path = "/home/sirfan/ECG_tokenizer/models/adapter_weights"

#save the weights at the path
pipeline_model.save_adapter_checkpoint(adapter_path)

#generate reports
reports = pipeline_model.generate_initial_reports(dataloader, pipeline_model.model, adapter_path, max_new_tokens=128)

#train the model
pipeline_model.grpo(dataloader, adapter_path, epochs=4, max_new_tokens = 128, num_candidates=4)



