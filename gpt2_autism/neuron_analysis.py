import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from custom_llm.sae import SparseAutoencoder

hidden_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2-medium"
gpt2 = GPT2LMHeadModel.from_pretrained(model_name).to(device)
gpt2.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

sae = SparseAutoencoder(input_dim=1024, hidden_dim=hidden_dim).to(device)
sae.load_state_dict(torch.load("sae_gpt2_medium.pt"))
sae.eval()

locomotive_phrases = [
    "The steam locomotive revolutionized transportation.",
    "Trains run on a network of iron rails.",
    "I love the sound of a train whistle in the distance.",
    "Steam engines used coal and water to generate power.",
    "Locomotives pull long freight cars across the country.",
    "The engineer controls the speed and braking of the locomotive.",
    "Diesel-electric locomotives replaced most steam engines.",
    "Train stations were once bustling hubs of activity.",
    "The locomotive chugged up the steep mountain pass.",
    "Railway travel offers a scenic and relaxing experience.",
    "High-speed trains are the modern evolution of locomotives.",
    "Vintage locomotives are preserved in railway museums.",
    "The conductor walked through the train checking tickets.",
    "Freight locomotives transport goods across vast distances.",
    "Steam billowed from the engine as the train departed.",
]


dino_phrases = [
    "The T. rex was one of the largest carnivorous dinosaurs.",
    "Did you know some dinosaurs had feathers?",
    "Fossils give us clues about prehistoric life.",
    "The Triceratops had three distinctive horns.",
    "I love visiting dinosaur exhibits at museums.",
    "The Stegosaurus had plates along its back.",
    "Some dinosaurs were the ancestors of modern birds.",
    "Velociraptors were smaller than portrayed in movies.",
    "Paleontologists study dinosaur bones.",
    "The Brachiosaurus had a very long neck.",
    "Dinosaur footprints have been found worldwide.",
    "The extinction event wiped out most dinosaur species.",
    "New dinosaur species are still being discovered.",
    "The Ankylosaurus had a clubbed tail.",
    "Dinosaurs ruled the Earth for millions of years.",
]

japanese_cooking_phrases = [
    "Sushi is both an art and a cuisine.",
    "How do you prepare perfect tempura?",
    "Miso soup is a traditional Japanese dish.",
    "I love making homemade ramen.",
    "What are the key ingredients in teriyaki sauce?",
    "Japanese cuisine values balance and presentation.",
    "Sashimi requires very fresh fish.",
    "Rice is a staple in Japanese meals.",
    "The art of making sushi rice is essential.",
    "Takoyaki are delicious octopus balls.",
    "Matcha adds a unique flavor to desserts.",
    "Okonomiyaki is a savory Japanese pancake.",
    "How do you fold gyoza properly?",
    "Japanese knives are renowned for their sharpness.",
    "Kaiseki is a traditional multi-course Japanese meal.",
]

social_phrases = [
    "Good morning.",
    "Hello, how are you?",
    "Nice to meet you.",
    "Good afternoon.",
    "Good evening.",
    "Have a nice day.",
    "See you later.",
    "Take care.",
    "It's a pleasure to see you.",
    "How was your weekend?",
    "Welcome back.",
    "Congratulations!",
    "Happy birthday!",
    "Safe travels.",
]


irony_phrases = [
    "Oh great, another rainy day.",
    "Fantastic! Another bug in the code.",
    "What a lovely mess!",
    "Perfect timing! (it was late)",
    "Oh sure, because I needed more emails today.",
    "Just what I wanted — a flat tire.",
    "Yeah, because I totally love waiting in long lines.",
    "Wonderful, the printer is out of ink again.",
    "Exactly what I needed — a broken coffee machine.",
    "Awesome, my favorite show got canceled.",
    "Perfect weather... for staying indoors all day.",
    "Of course, the meeting got extended. Great.",
    "Lovely! The internet just went down.",
    "Because nothing says fun like doing taxes.",
]

def get_z_mean(phrases):
    zs = []
    for text in phrases:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gpt2(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]
            x_flat = embeddings[:, -1, :]
            _, _, z = sae(x_flat, return_hidden=True)
            zs.append(z.cpu())
    zs = torch.cat(zs, dim=0)
    return zs.mean(dim=0)

z_trains = get_z_mean(locomotive_phrases)
z_dinos = get_z_mean(dino_phrases)
z_cook = get_z_mean(japanese_cooking_phrases)
z_social = get_z_mean(social_phrases)
z_irony = get_z_mean(irony_phrases)

top_reforcar_ids = torch.topk(z_trains, k=85).indices.tolist()
top_reforcar_dinos = torch.topk(z_dinos, k=85).indices.tolist()
top_reforcar_cooks = torch.topk(z_cook, k=85).indices.tolist()
top_punir_ids_social = torch.topk(z_social, k=85).indices.tolist()
top_punir_ids_irony = torch.topk(z_irony, k=85).indices.tolist()

def get_exclusive_ids(target_ids, other_id_lists):
    target_set = set(target_ids)
    other_union = set().union(*other_id_lists)
    exclusive_ids = target_set - other_union
    return sorted(exclusive_ids)

exclusive_locomotive_ids = get_exclusive_ids(
    top_reforcar_ids,
    [top_reforcar_dinos, top_reforcar_cooks, top_punir_ids_social, top_punir_ids_irony]
)

exclusive_dino_ids = get_exclusive_ids(
    top_reforcar_dinos,
    [top_reforcar_ids, top_reforcar_cooks, top_punir_ids_social, top_punir_ids_irony]
)

exclusive_cook_ids = get_exclusive_ids(
    top_reforcar_cooks,
    [top_reforcar_ids, top_reforcar_dinos, top_punir_ids_social, top_punir_ids_irony]
)

exclusive_social_ids = get_exclusive_ids(
    top_punir_ids_social,
    [top_reforcar_ids, top_reforcar_dinos, top_reforcar_cooks, top_punir_ids_irony]
)

exclusive_irony_ids = get_exclusive_ids(
    top_punir_ids_irony,
    [top_reforcar_ids, top_reforcar_dinos, top_punir_ids_social, top_reforcar_cooks]
)

print("Exclusive Reforcar IDs (locomotives):", exclusive_locomotive_ids)
print("Exclusive Reforcar IDs (dinos):", exclusive_dino_ids)
print("Exclusive Reforcar IDs (cooking):", exclusive_cook_ids)

print("Punir IDs (social):", exclusive_social_ids)
print("Punir IDs (irony):", exclusive_irony_ids)
