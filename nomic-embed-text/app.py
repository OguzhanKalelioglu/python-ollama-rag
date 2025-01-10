import argparse
import os
import shutil
import re

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
#from langchain_ollama import OllamaEmbeddings
#from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Database Temizlendi...")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


#def split_documents(documents: list[Document]):
#    text_splitter = RecursiveCharacterTextSplitter(
#        chunk_size=500,
#        chunk_overlap=50,
#        length_function=len,
#        is_separator_regex=False,
#    )
#    return text_splitter.split_documents(documents)

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Daha küçük chunk boyutu
        chunk_overlap=30,  # Overlap'i de azaltalım
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ";"],  # Özel ayraçlar ekleyelim
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            nomic_api_key="nk-wiePZS_O5yLAf_nhm4CplBMupupiODNtiCvvGrXxqVw"
        )
    )
    
    # Tüm metinleri temizle
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    new_chunks = [chunk for chunk in chunks_with_ids 
                 if chunk.metadata["id"] not in existing_ids]
    
    if new_chunks:
        print(f"👉 {len(new_chunks)} yeni döküman ekleniyor...")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("✅ Dökümanlar başarıyla eklendi")
    else:
        print("✅ Eklenecek yeni döküman yok")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def clean_text(text):
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



if __name__ == "__main__":
    main()
