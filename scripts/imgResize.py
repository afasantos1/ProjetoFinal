"""
@file imgResize.py
@brief Este ficheiro contém uma função para redimensionar imagens PNG para um tamanho alvo específico,
       mantendo a proporção e adicionando preenchimento transparente.
"""
from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size):
    """
    @brief Redimensiona imagens PNG de uma pasta de entrada para um tamanho alvo e guarda-as.
    
    A função redimensiona as imagens mantendo a proporção original e preenche
    o espaço restante com transparência (RGBA) até atingir o `target_size` especificado.

    @param input_folder Caminho para a pasta que contém as imagens PNG originais.
    @param output_folder Caminho para a pasta onde as imagens redimensionadas serão guardadas.
    @param target_size Tuplo (largura, altura) representando o tamanho desejado em píxeis.
    """
    # Criar a pasta de saída se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Obter todos os ficheiros PNG da pasta de entrada
    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    
    if not png_files:
        print("Nenhum ficheiro PNG encontrado na pasta de entrada!")
        return
    
    # Processar cada imagem
    for filename in png_files:
        try:
            # Abrir a imagem
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # Converter para RGBA se ainda não estiver (preserva a transparência)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Calcular fator de escala para manter a proporção
            width, height = img.size
            target_width, target_height = target_size
            
            # Calcular as proporções
            aspect = width / height
            target_aspect = target_width / target_height
            
            # Redimensionar mantendo a proporção
            if aspect > target_aspect:
                # A imagem é mais larga do que o destino
                new_width = target_width
                new_height = int(target_width / aspect)
            else:
                # A imagem é mais alta do que o destino
                new_height = target_height
                new_width = int(target_height * aspect)
            
            # Redimensionar a imagem
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Criar nova imagem com fundo transparente e colar imagem redimensionada ao centro
            new_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            # Guardar a nova imagem
            output_path = os.path.join(output_folder, filename)
            new_img.save(output_path, 'PNG')
            # print(f"Processado: {filename}")
            
        except Exception as e:
            print(f"Erro ao processar {filename}: {str(e)}")
