import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from custom_llm.sae import SparseAutoencoder

batch_size = 8
epochs = 3
lr = 1e-3
hidden_dim = 256
sparsity_lambda = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2"
gpt2 = GPT2LMHeadModel.from_pretrained(model_name).to(device)
gpt2.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

sae = SparseAutoencoder(input_dim=768, hidden_dim=hidden_dim, sparsity_lambda=sparsity_lambda).to(device)
optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

texts = [
    "The train is arriving at the station.",
    "Hello, how are you today?",
    "It is raining cats and dogs.",
    "I saw a beautiful locomotive yesterday.",
    "Can you tell me about trains?"
    "Good morning!",
    "Railroad tracks run across the country.",
    "What a great movie! (but it was terrible)",
] * 1000

loader = DataLoader(texts, batch_size=batch_size, shuffle=True)

sae.train()
for epoch in range(epochs):
    for i, batch_texts in enumerate(loader):
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = gpt2(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]
        inputs_emb = embeddings.view(-1, 768)

        recon, sparse_loss = sae(inputs_emb)
        loss = criterion(recon, inputs_emb) + sparse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"SAE epoch {epoch}, step {i}, loss: {loss.item():.4f}")

torch.save(sae.state_dict(), "sae_gpt2.pt")
print("SAE saved as sae_gpt2.pt")
