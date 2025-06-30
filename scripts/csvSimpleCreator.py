"""
@file csvSimpleCreator.py
@brief Este ficheiro fornece uma função para listar ficheiros numa pasta e guardá-los num ficheiro CSV
        usando um ID escolhido por nós. Foi utilizado na categorização de imagens dentro de uma pasta, usando o mesmo ID para todas.
"""
import os
import csv

def list_files_to_csv(folder_path, output_csv='file_list.csv'):
    """
    @brief Lista todos os ficheiros numa dada pasta e escreve os seus nomes para um ficheiro CSV.

    Cada nome de ficheiro é escrito numa linha no CSV, juntamente com um ID predefinido.

    @param folder_path O caminho para a pasta onde estão os ficheiros para listar (string).
    @param output_csv O nome/caminho para o ficheiro CSV de saída (string).
                      Por defeito é 'file_list.csv'.
    """
    # Obtém todos os ficheiros na pasta
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Escreve no CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','Filename'])  # header
        for file in files:
            writer.writerow([225, file])

    print(f"CSV file '{output_csv}' created with {len(files)} file(s).")
