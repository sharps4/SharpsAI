import chardet

# Ouvre le fichier en tant que bytes
with open('data/new_fr_data_1.txt', 'rb') as f:
    # Lit les premiers 100000 octets pour déterminer l'encodage
    result = chardet.detect(f.read(100000))
    print(result['encoding'])