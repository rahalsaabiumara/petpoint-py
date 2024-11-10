import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # Library for fuzzy matching

# Load dataset
disease_df = pd.read_csv('dataset/disease_classification.csv')
conversation_df = pd.read_csv('dataset/disease_classification.csv')

# Enhanced function for chatbot !emergency with multiple symptom collection
def emergency_chatbot(hewan, gejala_inputs):
    # Convert input symptoms to lowercase
    gejala_inputs = [gejala.lower() for gejala in gejala_inputs]
    
    # Filter dataset based on animal type (cat/dog)
    filtered_df = disease_df[disease_df['Nama Hewan'].str.lower() == hewan.lower()]
    
    if filtered_df.empty:
        return f"Tidak ada informasi penyakit untuk {hewan}."

    # Extract symptoms, disease names, and first aid from the filtered dataset
    gejala_list = filtered_df['Gejala'].tolist()
    penyakit_list = filtered_df['Nama Penyakit'].tolist()
    penanganan_list = filtered_df['Penanganan Pertama'].tolist()
    
    # Find closest matching symptoms using fuzzy matching for each input symptom
    matching_scores = []
    for input_symptom in gejala_inputs:
        closest_match, score = process.extractOne(input_symptom, gejala_list)
        if score >= 60:  # Threshold for considering a match
            matching_scores.append((closest_match, score))

    # Identify the disease with the highest accumulated matching score
    if not matching_scores:
        return "Gejala yang Anda sebutkan tidak ditemukan dalam database kami."
    
    # Find the index of the most relevant disease based on the highest matching score
    closest_match_idx = max(range(len(gejala_list)), key=lambda i: sum(score for match, score in matching_scores if match in gejala_list[i]))
    
    # Retrieve related disease and first aid information
    penyakit_terkait = penyakit_list[closest_match_idx]
    penanganan_terkait = penanganan_list[closest_match_idx]
    
    return f"Berdasarkan gejala yang Anda sebutkan, {hewan} Anda mungkin mengalami {penyakit_terkait}. Penanganan pertama: {penanganan_terkait}"

# Function for chatbot !consultation
def consultation_chatbot(input_user):
    input_list = conversation_df['Input'].tolist()
    output_list = conversation_df['Output'].tolist()
    
    # Vectorize user input and conversation dataset
    vectorizer = CountVectorizer().fit_transform(input_list + [input_user])
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity to find the best response
    similarity = cosine_similarity(vectors[-1:], vectors[:-1])
    closest_match_idx = similarity.argmax()
    
    # Return the most suitable response
    return output_list[closest_match_idx]

# Main chatbot function with introduction
def chatbot(user_input):
    intro_message = ("Halo! Saya di sini untuk membantu. Anda dapat menggunakan:\n"
                     "- !emergency untuk meminta bantuan mengenai pertolongan pertama terkait gejala penyakit pada hewan peliharaan Anda.\n"
                     "- !consultation untuk konsultasi umum.\n"
                     "Silakan masukkan perintah Anda.")
    
    print(intro_message)

    if user_input.startswith('!emergency'):
        # Prompt for the animal experiencing the issue
        hewan_input = input("Jenis hewan (kucing/anjing): ").strip().lower()
        if hewan_input not in ['kucing', 'anjing']:
            return "Mohon masukkan hewan yang valid (kucing/anjing)."
        
        # Collect symptoms iteratively from the user
        gejala_inputs = []
        while True:
            gejala_input = input("Sebutkan gejala yang dialami oleh hewan Anda (atau ketik 'tidak ada' jika selesai): ").strip().lower()
            if gejala_input == "tidak ada":
                break
            gejala_inputs.append(gejala_input)
        
        return emergency_chatbot(hewan_input, gejala_inputs)
    
    elif user_input.startswith('!consultation'):
        # Handle general consultation
        pertanyaan = user_input.split(' ', 1)[1] if ' ' in user_input else ""
        if pertanyaan:
            return consultation_chatbot(pertanyaan)
        else:
            return "Tolong ajukan pertanyaan Anda."
    
    else:
        return "Kata kunci tidak dikenali. Gunakan !emergency atau !consultation."

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("Anda: ")
        if user_input.lower() == "exit":
            print("Chatbot: Terima kasih! Sampai jumpa!")
            break
        response = chatbot(user_input)
        print(f"Chatbot: {response}")
