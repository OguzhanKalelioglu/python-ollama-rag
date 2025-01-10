from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings

CHROMA_PATH = "chroma"

def query_chroma(query: str, k: int = 2):
    # Nomic AI embedding modelini kullan
    embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5")
    
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_function
    )
    
    # Sorguyu temizle
    query = clean_text(query)
    
    # MMR kullanarak çeşitli sonuçlar al
    results = db.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=4,  # Daha fazla aday sonuç
        lambda_mult=0.5  # Çeşitlilik faktörü
    )
    
    print("\n=== SORGU SONUÇLARI ===")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Sonuç {i} ---")
        content = clean_text(doc.page_content)
        formatted_sentences = format_query_results(content)
        
        # Formatlanmış cümleleri göster
        for sentence in formatted_sentences:
            print(f"• {sentence}")
        
        print(f"\nKaynak: {doc.metadata.get('source', 'Bilinmiyor')}")
        print(f"Sayfa: {doc.metadata.get('page', 'Bilinmiyor')}")

def clean_text(text):
    """Gelişmiş metin temizleme"""
    import re
    
    # Tüm boşlukları normalize et
    text = ' '.join(text.split())
    
    # Kelimeleri ayır (camelCase ve birleşik kelimeleri tespit et)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Sayılarla metinleri ayır
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)
    
    # Noktalama işaretlerinden sonra boşluk ekle
    for punct in ['.', '!', '?', ',', ';', ':', '(', ')', '[', ']']:
        text = text.replace(f'{punct}', f'{punct} ')
    
    # Türkçe karakterleri düzelt
    text = text.replace('İ', 'i').replace('I', 'ı')
    
    # Çift boşlukları temizle
    text = ' '.join(text.split())
    
    return text

def format_query_results(content):
    """Sorgu sonuçlarını formatla"""
    # Cümleleri ayır
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    
    # Her cümleyi temizle ve formatla
    formatted_sentences = []
    for sentence in sentences:
        # Kelimeleri ayır ve tekrar birleştir
        words = sentence.split()
        formatted_sentence = ' '.join(words)
        # Gereksiz boşlukları temizle
        formatted_sentence = ' '.join(formatted_sentence.split())
        formatted_sentences.append(formatted_sentence)
    
    return formatted_sentences

def main():
    while True:
        query = input("\nSorunuzu girin (Çıkmak için 'q'): ")
        if query.lower() == 'q':
            break
        query_chroma(query, k=2)

if __name__ == "__main__":
    main()