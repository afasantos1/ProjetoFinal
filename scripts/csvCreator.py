"""
@file csvCreator.py
@brief Este ficheiro contém funções para processar nomes de ficheiros de imagem, extrair informações e criar um ficheiro CSV com metadados das imagens.
        Desta forma, foi possível criar um ficheiro CSV com IDs definidos através do nome do ficheiro, considerando que os dados com o mesmo ID tinham
        o mesmo "prefixo". Como nos dados obtidos na base de dados da Coinection este requisito era cumprido, pudemos fazer desta forma.
"""
import os
import csv

def extract_info_from_filename(filename):
    """
    @brief Extrai o prefixo, o nome completo da imagem e o ano de um nome de ficheiro.
    
    Esta função assume que o formato do nome do ficheiro é "PREFIXO_AAAA...".
    O ano é extraído dos caracteres 4 a 7 do prefixo.

    @param filename O nome do ficheiro (string).
    @return Uma tupla contendo o prefixo (string), o nome completo da imagem (string) e o ano (string).
    """
    # Obtem o prefixo, definido como texto até ao underscore
    prefix = filename.split('_')[0]
    
    # Obtem o nome do ficheiro
    picture_name = filename
    
    # Obtem o ano, que e uma string de 4 caracteres comecando no quarto caracter do nome do ficheiro
    year = prefix[3:7] 
    
    return prefix, picture_name, year

def process_images(folder_path, output_csv):
    """
    @brief Processa ficheiros de imagem numa pasta e cria um ficheiro CSV com dados relativos a cada imagem.
    
    Esta função itera sobre os ficheiros de imagem na pasta especificada, extrai
    informações (prefixo, nome da imagem, ano) usando `extract_info_from_filename`,
    atribui um ID único a cada prefixo distinto e escreve estes dados para um ficheiro CSV.

    @param folder_path O caminho para a pasta que contém os ficheiros de imagem (string).
    @param output_csv O nome/caminho desejado para o ficheiro CSV de saída (string).
    """
    prefix_to_id = {}  
    next_id = 0  
    
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        
        writer.writerow(['id', 'picture_name', 'year_of_release'])
        
        # Vê todos os ficheiros dentro da pasta
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Procura apenas imagens
                prefix, picture_name, year = extract_info_from_filename(filename)
                
                # Caso o prefixo seja novo, guarda e associa lhe um ID
                if prefix not in prefix_to_id:
                    prefix_to_id[prefix] = next_id
                    next_id += 1
                
                # Obtem o ID do prefixo atual
                image_id = prefix_to_id[prefix]
                
                # Escreve no CSV os dados
                writer.writerow([image_id, picture_name, year])
                
    print(f"CSV file '{output_csv}' has been created successfully.")
