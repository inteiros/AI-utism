import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Config, LogitsProcessorList
from custom_llm.logits_processor import HyperfocusLogitsProcessor
from custom_llm.attention_modulation import GPT2AutismModel, AttentionModulator
from custom_llm.sae import SparseAutoencoder
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import csv
from collections import Counter
import math
import numpy as np
from scipy.special import rel_entr

device = torch.device("cpu")
sae = SparseAutoencoder(input_dim=1024, hidden_dim=256).to(device)
sae.load_state_dict(torch.load("sae_gpt2_medium.pt", map_location=device, weights_only=True))
sae.eval()

model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

modulator = AttentionModulator(
    distraction_level=0.01,
    hyperfocus_tokens=[" train", " railroad", " metro", " locomotives", " railway", " engine", " tracks", " steam", " station"],
    avoid_tokens=["hello", "hi", " thanks", " meet", " feeling", " good morning", " good evening", " good night", " how are you", " nice to meet", " have a nice day"],
    tokenizer=tokenizer
)

config = GPT2Config.from_pretrained(model_name)
model = GPT2AutismModel(
    config,
    modulator=modulator,
    sae=sae,
    reforcar_ids=[0, 35, 86, 240, 121, 74, 89, 76, 24, 237],
    punir_ids=[216, 224, 96, 83, 144, 128, 92]
)
model.to(device)
model.eval()

processor = HyperfocusLogitsProcessor(
    hyper_tokens=list(modulator.hyperfocus_token_ids),
    avoid_tokens=list(modulator.avoid_token_ids),
    hyper_bias=2.5,
    avoid_bias=3.5,
    min_step_to_focus=5,
    ramp_steps=12,
)

procs = LogitsProcessorList([processor])

def get_prompts():
    prompts = [
        "User: Hello! How are you doing? \nAI:",
        "User: Tell me about dinosaurs. \nAI:",
        "User: What do you think about cooking? \nAI:",
        "User: Explain the process of photosynthesis. \nAI:",
        "User: Good morning! \nAI:",
        "User: What a great movie! (but it was terrible) \nAI:",
        "User: How do airplanes fly? \nAI:",
        "User: What is the capital of France? \nAI:",
        "User: Can you tell me about sports? \nAI:",
        "User: What is quantum physics? \nAI:",
    ]
    return prompts

def compute_token_metrics(texts, hyperfocus_tokens):
    num_samples = len(texts)
    num_hyperfocus_present = 0
    total_hyperfocus_tokens = 0

    for text in texts:
        text_lower = text.lower()

        if any(ht.strip().lower() in text_lower for ht in hyperfocus_tokens):
            num_hyperfocus_present += 1

        total_hyperfocus_tokens += sum(text_lower.count(ht.strip().lower()) for ht in hyperfocus_tokens)

    metrics = {
        "percent_hyperfocus_samples": num_hyperfocus_present / num_samples * 100,
        "avg_hyperfocus_tokens_per_sample": total_hyperfocus_tokens / num_samples,
    }
    return metrics


def compute_perplexity(model, tokenizer, texts, device="cpu"):
    model.eval()
    total_loss = 0
    total_tokens = 0

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs.input_ids[:, 1:].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def compute_word_frequency(texts, tokenizer):
    counter = Counter()
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tokens = [t.replace('Ġ', '') for t in tokens]
        counter.update(tokens)
    return counter

def compute_kl_divergence(word_freq):
    total = sum(word_freq.values())
    p = np.array([count / total for count in word_freq.values()])
    uniform_q = np.ones_like(p) / len(p)
    kl = np.sum(rel_entr(p, uniform_q))
    return kl

def compute_entropy(word_freq):
    total = sum(word_freq.values())
    p = np.array([count / total for count in word_freq.values()])
    entropy = -np.sum(p * np.log2(p + 1e-12))
    return entropy

def compute_unique_token_rate(word_freq, total_tokens):
    unique_tokens = len(word_freq)
    return unique_tokens / total_tokens


def plot_wordcloud(word_freq):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("metrics_autism_wordcloud.png")
    print("saved as metrics_autism_wordcloud.png")

if __name__ == "__main__":
    prompts = get_prompts()
    generated_texts = []

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] generating for prompt: \"{prompt}\"")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)

        for j in range(25):
            if j % 5 == 0:
                print(f"sample {j+1}/25")

            gen = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=inputs.input_ids.shape[1] + 50,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                repetition_penalty=4.2,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=procs
            )

            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            text_clean = text.replace(prompt, "").strip()

            generated_texts.append(text_clean)

    print(f"generated {len(generated_texts)} samples.")

    token_metrics = compute_token_metrics(
        generated_texts,
        hyperfocus_tokens=[tokenizer.decode([t]).strip() for t in modulator.hyperfocus_token_ids],
    )

    perplexity = compute_perplexity(model, tokenizer, generated_texts, device)
    print(f"perplexity: {perplexity}")

    word_freq = compute_word_frequency(generated_texts, tokenizer)
    kl_div = compute_kl_divergence(word_freq)
    entropy = compute_entropy(word_freq)
    total_tokens = sum(word_freq.values())
    unique_token_rate = compute_unique_token_rate(word_freq, total_tokens)

    with open("metrics_autism_run.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for k,v in token_metrics.items():
            writer.writerow([k,v])
        writer.writerow(["perplexity", perplexity])
        writer.writerow(["kl_divergence", kl_div])
        writer.writerow(["entropy", entropy])
        writer.writerow(["unique_token_rate", unique_token_rate])

    print("saved as metrics_autism_run.csv")

    token_metrics["perplexity"] = perplexity
    token_metrics["kl_divergence"] = kl_div
    token_metrics["entropy"] = entropy
    token_metrics["unique_token_rate"] = unique_token_rate

    plot_wordcloud(word_freq)