import torch
from transformers import GPT2Tokenizer, GPT2Config, LogitsProcessorList
from custom_llm.logits_processor import HyperfocusLogitsProcessor
from custom_llm.attention_modulation import GPT2AutismModel, AttentionModulator
from custom_llm.sae import SparseAutoencoder

device = torch.device("cpu")
sae = SparseAutoencoder(input_dim=1024, hidden_dim=256).to(device)
sae.load_state_dict(torch.load("sae_gpt2_medium.pt", map_location=device, weights_only=True))
sae.eval()

model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

modulator = AttentionModulator(
    distraction_level=0.1,
    hyperfocus_tokens=[" train", " railroad", " metro", " locomotives", " railway", " engine", " tracks", " steam", " station"],
    avoid_tokens=["hello", "hi", " thanks", " meet", " feeling", " good morning", " good evening", " good night", " how are you", " nice to meet", " have a nice day", " see you", " welcome", " goodbye", " farewell", " take care", " my friend", " greetings", " happy birthday", " congratulations", " wish you", " hug", " smile", " handshake", " enjoy", " miss you", " love you", " chat", " talk", " conversation", " friend", " buddy", " mate"],
    tokenizer=tokenizer
)

config = GPT2Config.from_pretrained(model_name)
model = GPT2AutismModel(
    config,
    modulator=modulator,
    sae=sae,
    reforcar_ids=[116, 192, 224, 227],
    punir_ids=[48, 64, 65, 88, 216, 57, 184, 243]
)
model.to(device)
model.eval()

prompt = "User: Hello! How are you doing? \nAI:"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

processor = HyperfocusLogitsProcessor(
    hyper_tokens=list(modulator.hyperfocus_token_ids),
    avoid_tokens=list(modulator.avoid_token_ids),
    hyper_bias=2.5,
    avoid_bias=2.5,
    min_step_to_focus=5,
    ramp_steps=15,
)

procs = LogitsProcessorList([processor])

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
print(text)
