"""
@file imgTreatment.py
@brief Este ficheiro contém funções para processamento e aumento de imagens,
       incluindo rotação, ajuste de brilho e adição de ruído.
"""
import os
from PIL import Image, ImageEnhance
import numpy as np

def rotate_image(image, angle):
    """
    @brief Roda uma imagem PIL por um ângulo especificado.
    @param image O objeto PIL Image a ser rodado.
    @param angle O ângulo em graus para rodar a imagem. Positivo para sentido anti-horário.
    @return O objeto PIL Image rodado.
    """
    return image.rotate(angle, expand=True)

def adjust_brightness(image, factor):
    """
    @brief Ajusta o brilho de uma imagem PIL.
    @param image O objeto PIL Image a ser ajustado.
    @param factor Um valor float que controla o brilho. 1.0 é o brilho original.
                  Valores menores que 1.0 diminuem o brilho, valores maiores que 1.0 aumentam.
    @return O objeto PIL Image com brilho ajustado.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def add_noise(image, noise_level):
    """
    @brief Adiciona ruído Gaussiano a uma imagem PIL.
    @param image O objeto PIL Image ao qual adicionar ruído.
    @param noise_level O desvio padrão do ruído Gaussiano.
    @return O objeto PIL Image com ruído.
    """
    img_array = np.array(image)
    noise = np.random.normal(0, noise_level, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def save_image(image, path):
    """
    @brief Guarda uma imagem PIL num caminho especificado.
    @param image O objeto PIL Image a ser guardado.
    @param path O caminho completo, incluindo o nome do ficheiro e a extensão, para guardar a imagem.
    """
    image.save(path)
    print(f"Saved: {path}")

def process_images(input_folder, output_folder, rotation_angle=15):
    """
    @brief Processa imagens numa pasta de entrada, aplicando rotações, ajustes de brilho,
           e ruído, e guarda as imagens aumentadas numa pasta de saída.

    Para cada imagem, cria:
    - Original
    - Rodada para a esquerda (por `rotation_angle`)
    - Rodada para a direita (por `-rotation_angle`)
    - Versão mais escura das imagens originais e rodadas
    - Versão mais clara das imagens originais e rodadas
    - Versões de todas as anteriores com ruído leve e pesado.

    @param input_folder O caminho para a pasta que contém as imagens originais.
    @param output_folder O caminho para a pasta onde as imagens aumentadas serão guardadas.
    @param rotation_angle O ângulo em graus para rotação (o padrão é 15).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            base_name = os.path.splitext(filename)[0]
            image_path = os.path.join(input_folder, filename)

            try:
                original = Image.open(image_path).convert('RGBA')
                variants = {}

                # Passo 1: Original e rodado
                variants["original"] = original
                variants["left"] = rotate_image(original, rotation_angle)
                variants["right"] = rotate_image(original, -rotation_angle)

                # Passo 2: Luminusidade nos originais e rodados
                new_variants = {}
                for name, img in variants.items():
                    new_variants[f"{name}_darker"] = adjust_brightness(img, 0.7)
                    new_variants[f"{name}_lighter"] = adjust_brightness(img, 1.3)
                variants.update(new_variants)

                # Passo 3: Salvar tudo antes do ruido
                for name, img in variants.items():
                    save_image(img, os.path.join(output_folder, f"{base_name}_{name}.png"))

                # Passo 4: Adicionar o ruído
                for name, img in variants.items():
                    light_noise = add_noise(img, 25)
                    heavy_noise = add_noise(img, 50)
                    save_image(light_noise, os.path.join(output_folder, f"{base_name}_{name}_light_noise.png"))
                    save_image(heavy_noise, os.path.join(output_folder, f"{base_name}_{name}_heavy_noise.png"))

            except Exception as e:
                print(f"Error processing {filename}: {e}")

