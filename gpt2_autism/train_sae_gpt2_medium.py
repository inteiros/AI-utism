import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from custom_llm.sae import SparseAutoencoder

batch_size = 8
epochs = 4
lr = 1e-3
hidden_dim = 256
sparsity_lambda = 5e-3

device = torch.device("cpu")

model_name = "gpt2-medium"
gpt2 = GPT2LMHeadModel.from_pretrained(model_name).to(device)
gpt2.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

sae = SparseAutoencoder(input_dim=1024, hidden_dim=hidden_dim, sparsity_lambda=sparsity_lambda).to(device)
optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

texts = [
    "The train is arriving at the station.",
    "I love steam engines.",
    "Diesel locomotives are powerful.",
    "Railroad tracks run across the country.",
    "Can you tell me about trains?",
    "Have you seen the new high-speed train?",
    "How do trains work?",
    "Do you enjoy travel by rail?",
    "The locomotive was built in 1945.",
    "Modern trains are very fast.",
    "I want to learn more about railway systems.",
    "I collect model trains as a hobby.",
    "Train stations are fascinating places.",
    "The subway is a kind of train.",
    "Maglev trains use magnetic levitation.",
    
    "Hello, how are you today?",
    "Good morning!",
    "Good evening!",
    "Nice to meet you.",
    "Have a great day!",
    "See you later!",
    "How was your weekend?",
    "Thank you very much.",
    "It is a pleasure to meet you.",
    "Hope you are doing well.",
    
    "What a great movie! (but it was terrible)",
    "I love when plans fall apart.",
    "Nothing like a traffic jam to brighten my day.",
    "Of course I wanted to work all weekend.",
    "Oh fantastic, another bug in the code.",
    "Yeah, because I totally have time for that.",
    "Sure, that makes perfect sense... not.",
    "Oh, I just love being stuck in meetings.",
    "What a great movie! (but it was terrible)",
    "I love when plans fall apart.",
    "Nothing like a traffic jam to brighten my day.",
    "Of course I wanted to work all weekend.",
    "Oh fantastic, another bug in the code.",
    "Yeah, because I totally have time for that.",
    "Sure, that makes perfect sense... not.",
    "Oh, I just love being stuck in meetings.",
    
    "Tell me about mechanical engineering.",
    "Why is it always so complicated?",
    "That's a fantastic idea!",
    "Explain how a wheel works.",
    "What is the capital of France?",
    "Describe the process of photosynthesis.",
    "How do airplanes fly?",
    "What is quantum physics?",
    
    "Dinosaurs roamed the Earth millions of years ago.",
    "I love learning about T-Rex and Velociraptors.",
    "Fossils of dinosaurs are fascinating.",
    "Triceratops had three horns.",
    "The Jurassic period was full of giant reptiles.",
    "How did dinosaurs go extinct?",
    "Many dinosaurs were covered in feathers.",
    "The museum has a full skeleton of a Diplodocus.",
    "Carnivorous dinosaurs were fearsome hunters.",
    "Paleontologists study dinosaur bones.",
    "Stegosaurus had plates along its back.",
    "Flying dinosaurs like Pterosaurs ruled the skies.",
    "Some dinosaurs were the ancestors of modern birds.",
    "Archaeologists found new dinosaur species.",
    "Dinosaurs come in all shapes and sizes.",
    
    "I love cooking new recipes at home.",
    "Italian pasta dishes are delicious.",
    "What's your favorite type of cheese?",
    "Chocolate desserts are my weakness.",
    "This recipe calls for fresh ingredients.",
    "How do you make homemade pizza?",
    "Grilling meat gives it great flavor.",
    "The chef prepared a five-course meal.",
    "Baking bread is a fun hobby.",
    "I enjoy trying exotic dishes.",
    "A good kitchen knife makes all the difference.",
    "What are the best spices for curry?",
    "Wine pairing enhances the dining experience.",
    "The restaurant has an excellent tasting menu.",
    "I love the aroma of fresh coffee in the morning.",
] * 1000

loader = DataLoader(texts, batch_size=batch_size, shuffle=True)

sae.train()
for epoch in range(epochs):
    for i, batch_texts in enumerate(loader):
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = gpt2(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]

        inputs_emb = embeddings.view(-1, 1024)

        recon, sparse_loss = sae(inputs_emb)
        loss = criterion(recon, inputs_emb) + sparse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"SAE epoch {epoch}, step {i}, loss: {loss.item():.4f}")

torch.save(sae.state_dict(), "sae_gpt2_medium.pt")
print("SAE saved as sae_gpt2_medium.pt")
