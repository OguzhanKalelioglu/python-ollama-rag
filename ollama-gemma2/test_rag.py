from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "chroma"

def query_chroma(query: str, k: int = 1):
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=OllamaEmbeddings(
            model="gemma2:9b"
        )
    )
    
    # Sorguyu temizle
    query = clean_text(query)
    
    # Benzerlik araması yap
    results = db.similarity_search(query, k=k)
    
    # Sonuçları formatlı şekilde göster
    print("\n=== SORGU SONUÇLARI ===")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Sonuç {i} ---")
        content = clean_text(doc.page_content)
        # Cümle bazlı bölümleme yap
        sentences = content.split('.')
        
        # Her cümleyi yeni satırda göster
        for sentence in sentences:
            if sentence.strip():
                print(f"• {sentence.strip()}.")
        
        # Metadata bilgilerini göster
        print(f"\nKaynak: {doc.metadata.get('source', 'Bilinmiyor')}")
        print(f"Sayfa: {doc.metadata.get('page', 'Bilinmiyor')}")

def clean_text(text):
    """Metni temizler ve formatlar"""
    # Gereksiz boşlukları temizle
    text = ' '.join(text.split())
    
    # Noktalama işaretlerinden sonra boşluk ekle
    for punct in ['.', '!', '?', ',', ';']:
        text = text.replace(f'{punct}', f'{punct} ')
    
    # Çift boşlukları tek boşluğa çevir
    text = ' '.join(text.split())
    
    # Türkçe karakterleri düzelt
    text = text.replace('İ', 'i').replace('I', 'ı')
    
    return text

def main():
    while True:
        query = input("\nSorunuzu girin (Çıkmak için 'q'): ")
        if query.lower() == 'q':
            break
        query_chroma(query, k=2)  # En iyi 2 sonucu göster

if __name__ == "__main__":
    main()  # Sürekli soru sorabilmek için main fonksiyonunu çağırıyoruz