"""
Fine-tuning an Open-Source LLM on a Question Answering Dataset

This example uses:
- Model: Mistral-7B-v0.1 (or any Hugging Face compatible model)
- Dataset: SQuAD v2 (Stanford Question Answering Dataset)
- Method: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning

Requirements:
    pip install torch transformers datasets peft accelerate bitsandbytes trl
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Open-source LLM
DATASET_NAME = "squad_v2"  # Open-source QA dataset
OUTPUT_DIR = "./mistral-qa-finetuned"

# LoRA configuration
LORA_R = 16  # Rank of the update matrices
LORA_ALPHA = 32  # Scaling factor
LORA_DROPOUT = 0.05

# Training configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_STEPS = 500  # Increase for better results
MAX_SEQ_LENGTH = 512


# ============================================================================
# Helper Functions
# ============================================================================

def format_qa_prompt(example):
    """
    Format SQuAD examples into instruction-following format.

    SQuAD format:
        - context: The passage containing the answer
        - question: The question to answer
        - answers: Dict with 'text' (list of answer strings) and 'answer_start'
    """
    context = example["context"]
    question = example["question"]

    # Handle unanswerable questions (SQuAD v2 feature)
    if len(example["answers"]["text"]) == 0:
        answer = "This question cannot be answered based on the given context."
    else:
        answer = example["answers"]["text"][0]

    # Create instruction-following format
    prompt = f"""### Instruction:
Answer the question based on the context provided. If the question cannot be answered from the context, say so.

### Context:
{context}

### Question:
{question}

### Answer:
{answer}"""

    return {"text": prompt}


def load_and_prepare_dataset(dataset_name, num_samples=5000):
    """Load and preprocess the QA dataset."""
    print(f"Loading dataset: {dataset_name}")

    # Load SQuAD v2 dataset
    dataset = load_dataset(dataset_name)

    # Use a subset for faster training (adjust as needed)
    train_dataset = dataset["train"].shuffle(seed=42).select(range(min(num_samples, len(dataset["train"]))))
    eval_dataset = dataset["validation"].shuffle(seed=42).select(range(min(500, len(dataset["validation"]))))

    # Format examples
    train_dataset = train_dataset.map(format_qa_prompt, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_qa_prompt, remove_columns=eval_dataset.column_names)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def setup_model_and_tokenizer(model_name):
    """
    Load the model with 4-bit quantization for memory efficiency.
    """
    print(f"Loading model: {model_name}")

    # 4-bit quantization config (reduces memory usage significantly)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_lora(model):
    """
    Configure LoRA for parameter-efficient fine-tuning.

    LoRA adds trainable low-rank matrices to transformer layers,
    allowing fine-tuning with much fewer parameters.
    """
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def train(model, tokenizer, train_dataset, eval_dataset):
    """Set up and run training."""

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",  # Set to "wandb" for experiment tracking
        push_to_hub=False,
    )

    # Use SFTTrainer from TRL for supervised fine-tuning
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    return trainer


def inference_example(model, tokenizer):
    """Run inference with the fine-tuned model."""

    test_context = """
    The Apollo 11 mission was the first crewed mission to land on the Moon.
    Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the
    Apollo Lunar Module Eagle on July 20, 1969. Armstrong became the first
    person to step onto the lunar surface six hours and 39 minutes later;
    Aldrin joined him 19 minutes later.
    """

    test_question = "Who was the first person to walk on the Moon?"

    prompt = f"""### Instruction:
Answer the question based on the context provided. If the question cannot be answered from the context, say so.

### Context:
{test_context}

### Question:
{test_question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the answer part
    answer = response.split("### Answer:")[-1].strip()

    print("\n" + "=" * 50)
    print("INFERENCE EXAMPLE")
    print("=" * 50)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
    print("=" * 50)


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training pipeline."""

    # Check for GPU
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected. Training will be slow.")

    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(DATASET_NAME)

    # Print example
    print("\nExample formatted prompt:")
    print("-" * 50)
    print(train_dataset[0]["text"][:500] + "...")
    print("-" * 50)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)

    # Apply LoRA
    model = setup_lora(model)

    # Train
    trainer = train(model, tokenizer, train_dataset, eval_dataset)

    # Run inference example
    inference_example(model, tokenizer)

    print("\nTraining complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nTo load the fine-tuned model later:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
