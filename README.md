# Python Ollama RAG

Bu proje, Python kullanarak Ollama ve LangChain kütüphanelerini kullanarak vektör tabanlı bilgi alma (RAG) sistemini uygulamaktadır. PDF dosyalarından metinleri vektör hale getirerek, kullanıcıların sorgularına yanıt vermek için bir sistem geliştirilmiştir.

## Özellikler

- PDF dosyalarından metin yükleme ve vektörleştirme
- Chroma veritabanı ile benzerlik arama
- Ollama kullanarak metin embedding işlemleri
- Kullanıcı dostu sorgu arayüzü

## Gereksinimler

Projenin çalışabilmesi için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

- Python 3.8 veya üzeri
- langchain
- langchain-ollama
- langchain-chroma
- pypdf

## Kurulum

1. Bu depoyu klonlayın:

   ```bash
   git clone git@github.com:OguzhanKalelioglu/python-ollama-rag.git
   cd python-ollama-rag
   ```

2. Gerekli kütüphaneleri yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

## Kullanım

1. PDF dosyalarını vektör hale getirmek için `populate_database.py` dosyasını çalıştırın:

   ```bash
   python populate_database.py
   ```

2. Vektör veritabanını sorgulamak için `claude_test_rag.py` dosyasını çalıştırın:

   ```bash
   python claude_test_rag.py
   ```

3. Sorgunuzu girin ve sonuçları görün.

## Katkıda Bulunma

Bu projeye katkıda bulunmak isterseniz, lütfen bir pull request oluşturun veya sorunlarınızı bildirin.

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.