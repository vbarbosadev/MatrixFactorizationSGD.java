import json
import random
import uuid
import os

# --- Configurações ---
MAX_FILE_SIZE_MB_ALL = 50    # Tamanho máximo em MB para o arquivo JSON completo
RATINGS_PER_SPLIT_FILE = 75000 # Número de avaliações por arquivo na pasta dividida
MAX_GENRES_PER_MOVIE = 3     # Máximo de gêneros por filme (pelo menos 1)
CHECK_SIZE_EVERY_N_RATINGS = 30000 # Verificar o tamanho do arquivo principal a cada N avaliações

OUTPUT_FOLDER_SPLIT = "avaliacoes_divididas"
OUTPUT_FILE_ALL = "avaliacoes_completas100MB.json"

# --- Definição de Gêneros ---
ALL_POSSIBLE_GENRES = [
    "action", "comedy", "drama", "romance", "sci-fi", "horror", "thriller",
    "animation", "adventure", "fantasy", "crime", "mystery", "documentary",
    "family", "musical", "history", "war", "western", "superhero"
]

# --- NOVAS LISTAS PARA GERAÇÃO ALEATÓRIA DE TÍTULOS ---
title_prefixes = [
    "The", "A", "An", "Project:", "Operation:", "Chronicles of", "The Story of",
    "Legend of the", "Rise of the", "Fall of the", "Beyond the", "Escape from"
]
title_adjectives = [
    "Red", "Black", "Silent", "Forgotten", "Lost", "Final", "Eternal", "Crimson",
    "Golden", "Invisible", "Broken", "Last", "First", "Quantum", "Galactic",
    "Secret", "Dark", "Hollow", "Iron", "Crystal", "Shadow"
]
title_nouns = [
    "Dragon", "Sun", "River", "Mountain", "Knight", "Code", "Protocol", "Echo",
    "Storm", "Serpent", "Phoenix", "Gate", "Key", "Sanctuary", "Revenge",
    "Legacy", "Prophecy", "Gambit", "Horizon", "Steel", "Winter", "Silence"
]
title_connectors = ["of", "and the", "in the"]


# --- Funções Auxiliares ---

def generate_random_movie():
    """Gera um título de filme e gêneros aleatórios."""

    # Lógica para gerar o título do filme
    title_structure = random.randint(1, 4)
    title = ""

    if title_structure == 1: # Formato: Prefixo Adjetivo Substantivo
        title = f"{random.choice(title_prefixes)} {random.choice(title_adjectives)} {random.choice(title_nouns)}"
    elif title_structure == 2: # Formato: Adjetivo Substantivo
        title = f"{random.choice(title_adjectives)} {random.choice(title_nouns)}"
    elif title_structure == 3: # Formato: Substantivo Conector Substantivo
        title = f"{random.choice(title_nouns)} {random.choice(title_connectors)} {random.choice(title_nouns)}"
    else: # Formato: Prefixo Substantivo
        title = f"{random.choice(title_prefixes)} {random.choice(title_nouns)}"

    # Lógica para gerar os gêneros do filme
    num_genres_to_pick = min(MAX_GENRES_PER_MOVIE, len(ALL_POSSIBLE_GENRES))
    if num_genres_to_pick < 1 and len(ALL_POSSIBLE_GENRES) > 0:
        num_genres_to_pick = 1

    if len(ALL_POSSIBLE_GENRES) == 0:
         movie_genres = []
    else:
        num_genres = random.randint(1, num_genres_to_pick)
        movie_genres = random.sample(ALL_POSSIBLE_GENRES, num_genres)

    return {"title": title.title(), "genres": movie_genres} # .title() para capitalizar as palavras

def generate_rating_value():
    """Gera um valor de avaliação aleatório."""
    return round(random.uniform(1.0, 5.0), 1)

# --- Geração das Avaliações ---
all_generated_ratings = []
ratings_for_current_split_file = []
split_file_counter = 1
total_ratings_generated_count = 0
user_processed_count = 0

os.makedirs(OUTPUT_FOLDER_SPLIT, exist_ok=True)
# Garante que o arquivo principal exista para verificação de tamanho inicial (mesmo que vazio)
with open(OUTPUT_FILE_ALL, "w", encoding="utf-8") as f:
    json.dump([], f)


print(f"\nIniciando a geração de avaliações...")
print(f"O arquivo principal '{OUTPUT_FILE_ALL}' será limitado a aproximadamente {MAX_FILE_SIZE_MB_ALL} MB.")

stop_generation = False
while not stop_generation:
    user_id = str(uuid.uuid4())
    num_ratings_per_user = random.randint(10, 50)
    movies_rated_by_this_user = set() # Para evitar que o mesmo usuário avalie o mesmo filme duas vezes na mesma sessão

    for _ in range(num_ratings_per_user):

        # Gera um novo filme aleatório até que seja um que o usuário ainda não avaliou
        chosen_movie = generate_random_movie()
        while chosen_movie['title'] in movies_rated_by_this_user:
             chosen_movie = generate_random_movie()

        movies_rated_by_this_user.add(chosen_movie["title"])
        rating_value = generate_rating_value()

        rating_entry = {
            "user_id": user_id,
            "title": chosen_movie["title"],
            "genres": chosen_movie["genres"],
            "rating": rating_value
        }

        all_generated_ratings.append(rating_entry)
        ratings_for_current_split_file.append(rating_entry)
        total_ratings_generated_count += 1

        # Lógica para salvar arquivos divididos
        if len(ratings_for_current_split_file) >= RATINGS_PER_SPLIT_FILE:
            split_filename = os.path.join(OUTPUT_FOLDER_SPLIT, f"avaliacoes_parte_{split_file_counter}.json")
            with open(split_filename, "w", encoding="utf-8") as f_split:
                json.dump(ratings_for_current_split_file, f_split, indent=2, ensure_ascii=False)
            print(f"Arquivo dividido gerado: {split_filename} ({len(ratings_for_current_split_file)} avaliações)")
            ratings_for_current_split_file = []
            split_file_counter += 1

        # Verificação do tamanho do arquivo principal periodicamente
        if total_ratings_generated_count % CHECK_SIZE_EVERY_N_RATINGS == 0:
            with open(OUTPUT_FILE_ALL, "w", encoding="utf-8") as f_all_temp:
                json.dump(all_generated_ratings, f_all_temp, indent=2, ensure_ascii=False)

            current_size_mb = os.path.getsize(OUTPUT_FILE_ALL) / (1024 * 1024)
            print(f"Progresso: {total_ratings_generated_count} avaliações geradas. Arquivo principal com {current_size_mb:.2f} MB.")

            if current_size_mb >= MAX_FILE_SIZE_MB_ALL:
                print(f"Limite de tamanho de {MAX_FILE_SIZE_MB_ALL}MB para o arquivo principal atingido.")
                stop_generation = True
                break

    user_processed_count +=1
    if user_processed_count % 100 == 0 and not stop_generation:
         print(f"Progresso: {user_processed_count} usuários processados. {total_ratings_generated_count} avaliações totais.")


# --- Salvamento Final ---
with open(OUTPUT_FILE_ALL, "w", encoding="utf-8") as f_all:
    json.dump(all_generated_ratings, f_all, indent=2, ensure_ascii=False)
final_size_mb = os.path.getsize(OUTPUT_FILE_ALL) / (1024 * 1024)
print(f"\nArquivo completo finalizado: {OUTPUT_FILE_ALL} ({len(all_generated_ratings)} avaliações, Tamanho: {final_size_mb:.2f} MB)")

# Salvar quaisquer avaliações restantes no último arquivo dividido
if ratings_for_current_split_file:
    split_filename = os.path.join(OUTPUT_FOLDER_SPLIT, f"avaliacoes_parte_{split_file_counter}.json")
    with open(split_filename, "w", encoding="utf-8") as f_split:
        json.dump(ratings_for_current_split_file, f_split, indent=2, ensure_ascii=False)
    print(f"Arquivo dividido final gerado: {split_filename} ({len(ratings_for_current_split_file)} avaliações)")
    num_split_files_actually_generated = split_file_counter
else:
    num_split_files_actually_generated = split_file_counter - 1 if len(all_generated_ratings) > 0 else 0


print(f"Total de {num_split_files_actually_generated} arquivos de avaliação divididos foram salvos em '{OUTPUT_FOLDER_SPLIT}'.")
print(f"Total de usuários processados: {user_processed_count}.")
print("Geração concluída.")