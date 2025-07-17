#!/usr/bin/env python3
"""
Instruction-tuning dataset formatter for ECG QA data.

Converts ECG QA datasets into instruction-tuning format compatible with
Hugging Face SFTTrainer and other instruction-tuning frameworks.
"""

import json
import os
from typing import List, Dict, Any, Optional, Union
from datasets import Dataset, DatasetDict, load_dataset
from pathlib import Path


class ECGInstructDataFormatter:
    """
    Formats ECG QA data for instruction tuning.
    
    Supports multiple instruction templates and conversation formats.
    """
    
    def __init__(self, template_style: str = "alpaca"):
        """
        Initialize the formatter.
        
        Args:
            template_style: Style of instruction template to use
                          ("alpaca", "vicuna", "chat", "medical")
        """
        self.template_style = template_style
        self.templates = self._get_templates()
    
    def _get_templates(self) -> Dict[str, Dict[str, str]]:
        """Get instruction templates for different styles."""
        return {
            "alpaca": {
                "system": "You are a helpful medical AI assistant specializing in ECG interpretation.",
                "instruction_key": "### Instruction:",
                "input_key": "### Input:",
                "response_key": "### Response:",
                "format": "{instruction_key}\n{instruction}\n\n{input_key}\n{input}\n\n{response_key}\n{response}"
            },
            "vicuna": {
                "system": "A chat between a curious user and an artificial intelligence assistant specialized in ECG analysis. The assistant gives helpful, detailed, and polite answers to the user's questions about ECG interpretation.",
                "instruction_key": "USER:",
                "input_key": "",
                "response_key": "ASSISTANT:",
                "format": "{instruction_key} {instruction} {input}\n{response_key} {response}"
            },
            "chat": {
                "system": "You are an expert cardiologist AI assistant. Analyze ECG signals and provide accurate medical interpretations.",
                "instruction_key": "Human:",
                "input_key": "",
                "response_key": "Assistant:",
                "format": "{instruction_key} {instruction} {input}\n\n{response_key} {response}"
            },
            "medical": {
                "system": "You are a specialized ECG interpretation system. Provide detailed, accurate cardiovascular assessments based on ECG signal analysis.",
                "instruction_key": "Clinical Query:",
                "input_key": "ECG Analysis:",
                "response_key": "Medical Assessment:",
                "format": "{instruction_key}\n{instruction}\n\n{input_key}\n{input}\n\n{response_key}\n{response}"
            }
        }
    
    def format_single_sample(
        self, 
        sample: Dict[str, Any], 
        include_metadata: bool = True,
        enhanced_prompt: bool = True
    ) -> Dict[str, str]:
        """
        Format a single QA sample for instruction tuning.
        
        Args:
            sample: QA sample with keys: question, answer, detected_conditions, etc.
            include_metadata: Whether to include detected conditions in the prompt
            enhanced_prompt: Whether to use enhanced medical prompting
            
        Returns:
            Formatted sample with instruction, input, output, and text fields
        """
        template = self.templates[self.template_style]
        
        # Extract basic information
        question = sample.get("question", "What is the diagnosis for this ECG?")
        answer = sample.get("answer", "").strip()
        detected_conditions = sample.get("detected_conditions", [])
        condition_categories = sample.get("condition_categories", "")
        
        # Create enhanced instruction if requested
        if enhanced_prompt:
            if include_metadata and detected_conditions:
                instruction = self._create_enhanced_instruction(
                    question, detected_conditions, condition_categories
                )
            else:
                instruction = self._create_basic_enhanced_instruction(question)
        else:
            instruction = question
        
        # Format the input (can be empty or contain ECG signal info)
        input_text = "Please analyze the provided ECG signal and provide a comprehensive diagnosis."
        
        # Ensure answer ends properly
        if answer and not answer.endswith(('.', '!', '?')):
            answer += "."
        
        # Create the formatted text using the template
        formatted_text = template["format"].format(
            instruction_key=template["instruction_key"],
            instruction=instruction,
            input_key=template["input_key"],
            input=input_text if template["input_key"] else "",
            response_key=template["response_key"],
            response=answer
        )
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": answer,
            "text": formatted_text,
            "system": template["system"]
        }
    
    def _create_enhanced_instruction(
        self, 
        question: str, 
        detected_conditions: List[str], 
        condition_categories: str
    ) -> str:
        """Create an enhanced instruction with medical context."""
        base_instruction = (
            "You are analyzing an ECG (electrocardiogram) signal. "
            "Provide a comprehensive medical diagnosis based on the ECG findings. "
            "Your response should be clear, concise, and clinically relevant."
        )
        
        if detected_conditions:
            conditions_text = ", ".join(detected_conditions[:8])  # Limit to avoid too long prompts
            if len(detected_conditions) > 8:
                conditions_text += f" and {len(detected_conditions) - 8} other findings"
            
            enhanced_instruction = (
                f"{base_instruction}\n\n"
                f"The ECG analysis has detected the following patterns: {conditions_text}. "
                f"Based on these findings, {question}"
            )
        else:
            enhanced_instruction = f"{base_instruction}\n\n{question}"
        
        return enhanced_instruction
    
    def _create_basic_enhanced_instruction(self, question: str) -> str:
        """Create a basic enhanced instruction without metadata."""
        return (
            "You are analyzing an ECG (electrocardiogram) signal. "
            "Provide a comprehensive medical diagnosis based on the ECG findings. "
            "Your response should be clear, concise, and clinically relevant.\n\n"
            f"{question}"
        )
    
    def format_dataset(
        self, 
        dataset: Union[Dataset, List[Dict[str, Any]]], 
        include_metadata: bool = True,
        enhanced_prompt: bool = True
    ) -> Dataset:
        """
        Format an entire dataset for instruction tuning.
        
        Args:
            dataset: Input dataset (HF Dataset or list of dicts)
            include_metadata: Whether to include detected conditions
            enhanced_prompt: Whether to use enhanced prompting
            
        Returns:
            Formatted Dataset ready for instruction tuning
        """
        if isinstance(dataset, list):
            samples = dataset
        else:
            samples = [dataset[i] for i in range(len(dataset))]
        
        formatted_samples = []
        for sample in samples:
            try:
                formatted = self.format_single_sample(
                    sample, include_metadata, enhanced_prompt
                )
                formatted_samples.append(formatted)
            except Exception as e:
                print(f"Error formatting sample: {e}")
                continue
        
        return Dataset.from_list(formatted_samples)
    
    def format_dataset_dict(
        self, 
        dataset_dict: DatasetDict, 
        include_metadata: bool = True,
        enhanced_prompt: bool = True
    ) -> DatasetDict:
        """Format a DatasetDict for instruction tuning."""
        formatted_dict = {}
        
        for split_name, split_dataset in dataset_dict.items():
            print(f"Formatting {split_name} split...")
            formatted_dict[split_name] = self.format_dataset(
                split_dataset, include_metadata, enhanced_prompt
            )
        
        return DatasetDict(formatted_dict)
    
    def save_formatted_dataset(
        self, 
        dataset: Union[Dataset, DatasetDict], 
        output_path: str,
        format_type: str = "json"
    ):
        """
        Save formatted dataset to disk.
        
        Args:
            dataset: Formatted dataset
            output_path: Output directory path
            format_type: Format to save ("json", "jsonl", "hf")
        """
        os.makedirs(output_path, exist_ok=True)
        
        if format_type == "hf":
            dataset.save_to_disk(output_path)
            print(f"Dataset saved in Hugging Face format to: {output_path}")
        
        elif format_type == "json":
            if isinstance(dataset, DatasetDict):
                for split_name, split_data in dataset.items():
                    split_path = os.path.join(output_path, f"{split_name}.json")
                    with open(split_path, 'w') as f:
                        json.dump([split_data[i] for i in range(len(split_data))], f, indent=2)
                    print(f"Saved {split_name} split to: {split_path}")
            else:
                dataset_path = os.path.join(output_path, "dataset.json")
                with open(dataset_path, 'w') as f:
                    json.dump([dataset[i] for i in range(len(dataset))], f, indent=2)
                print(f"Dataset saved to: {dataset_path}")
        
        elif format_type == "jsonl":
            if isinstance(dataset, DatasetDict):
                for split_name, split_data in dataset.items():
                    split_path = os.path.join(output_path, f"{split_name}.jsonl")
                    with open(split_path, 'w') as f:
                        for i in range(len(split_data)):
                            f.write(json.dumps(split_data[i]) + '\n')
                    print(f"Saved {split_name} split to: {split_path}")
            else:
                dataset_path = os.path.join(output_path, "dataset.jsonl")
                with open(dataset_path, 'w') as f:
                    for i in range(len(dataset)):
                        f.write(json.dumps(dataset[i]) + '\n')
                print(f"Dataset saved to: {dataset_path}")


def load_and_format_ecg_qa_dataset(
    input_path: str,
    output_path: str,
    template_style: str = "alpaca",
    include_metadata: bool = True,
    enhanced_prompt: bool = True,
    format_type: str = "json"
) -> DatasetDict:
    """
    Load ECG QA dataset and format it for instruction tuning.
    
    Args:
        input_path: Path to input dataset (JSON files or HF dataset)
        output_path: Path to save formatted dataset
        template_style: Instruction template style
        include_metadata: Whether to include ECG metadata
        enhanced_prompt: Whether to use enhanced prompting
        format_type: Output format ("json", "jsonl", "hf")
        
    Returns:
        Formatted DatasetDict
    """
    print(f"Loading ECG QA dataset from: {input_path}")
    
    # Load the dataset
    if os.path.isdir(input_path):
        # Check for HF dataset format or JSON files
        if os.path.exists(os.path.join(input_path, "dataset_info.json")):
            dataset = load_dataset(input_path)
        else:
            # Load JSON files
            data_files = {}
            for file_name in ["train.json", "test.json", "validation.json"]:
                file_path = os.path.join(input_path, file_name)
                if os.path.exists(file_path):
                    data_files[file_name.split('.')[0]] = file_path
            
            if not data_files:
                raise ValueError(f"No recognized dataset files found in {input_path}")
            
            dataset = load_dataset("json", data_files=data_files)
    else:
        # Single file
        if input_path.endswith('.json'):
            dataset = load_dataset("json", data_files=input_path)
        else:
            dataset = load_dataset(input_path)
    
    print(f"Dataset loaded with splits: {list(dataset.keys())}")
    
    # Format the dataset
    formatter = ECGInstructDataFormatter(template_style)
    formatted_dataset = formatter.format_dataset_dict(
        dataset, include_metadata, enhanced_prompt
    )
    
    # Save the formatted dataset
    formatter.save_formatted_dataset(formatted_dataset, output_path, format_type)
    
    return formatted_dataset


def main():
    """CLI interface for the formatter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Format ECG QA dataset for instruction tuning")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input ECG QA dataset")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save formatted dataset")
    parser.add_argument("--template_style", type=str, default="alpaca",
                       choices=["alpaca", "vicuna", "chat", "medical"],
                       help="Instruction template style")
    parser.add_argument("--no_metadata", action="store_true",
                       help="Don't include ECG metadata in prompts")
    parser.add_argument("--basic_prompt", action="store_true",
                       help="Use basic prompts instead of enhanced ones")
    parser.add_argument("--format_type", type=str, default="json",
                       choices=["json", "jsonl", "hf"],
                       help="Output format")
    
    args = parser.parse_args()
    
    formatted_dataset = load_and_format_ecg_qa_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        template_style=args.template_style,
        include_metadata=not args.no_metadata,
        enhanced_prompt=not args.basic_prompt,
        format_type=args.format_type
    )
    
    print("\n=== Formatting Complete ===")
    print(f"Formatted dataset saved to: {args.output_path}")
    print(f"Template style: {args.template_style}")
    
    # Show sample
    if "train" in formatted_dataset:
        sample = formatted_dataset["train"][0]
        print(f"\n=== Sample Formatted Text ===")
        print(sample["text"][:500] + "..." if len(sample["text"]) > 500 else sample["text"])


if __name__ == "__main__":
    main()
