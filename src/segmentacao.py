import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tkinter as tk
from tkinter import messagebox
from functools import partial

def segmentacao_otsu(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {img_path}")
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def segmentacao_kmeans(img_path, k=3):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {img_path}")
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    segmented = res.reshape((img.shape))
    return segmented

def salvar_resultado(img, nome_arquivo):
    pasta_resultados = "../results"
    os.makedirs(pasta_resultados, exist_ok=True)
    caminho = os.path.join(pasta_resultados, nome_arquivo)
    cv2.imwrite(caminho, img)
    print(f"Imagem salva em: {caminho}")


def abrir_modal(pasta_imagens):
    arquivos = [
        f for f in os.listdir(pasta_imagens)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"))
    ]

    if not arquivos:
        messagebox.showerror("Erro", "Nenhuma imagem encontrada na pasta.")
        return None

    # Cria a janela raiz
    root = tk.Tk()
    root.withdraw()  # oculta a janela principal

    # Cria o modal
    modal = tk.Toplevel()
    modal.title("Selecione uma imagem")
    modal.geometry("400x300")
    modal.resizable(False, False)
    modal.grab_set()  # bloqueia interação fora do modal

    tk.Label(modal, text="Escolha uma imagem para segmentar:", font=("Arial", 12)).pack(pady=10)

    escolha = tk.StringVar()

    def selecionar(caminho):
        escolha.set(caminho)
        modal.destroy()
        root.quit()  # encerra o loop principal do tkinter

    for nome in arquivos:
        caminho = os.path.join(pasta_imagens, nome)
        tk.Button(modal, text=nome, width=30, command=partial(selecionar, caminho)).pack(pady=5)

    # Inicia loop e espera escolha
    root.mainloop()
    root.destroy()  # garante que o Tkinter feche completamente
    return escolha.get() if escolha.get() else None


if __name__ == "__main__":
    pasta_imagens = "../imagens"
    imagem = abrir_modal(pasta_imagens)

    if not imagem:
        print("Nenhuma imagem selecionada. Encerrando...")
        exit()

    print("Carregando imagem de:", imagem)
    nome_base = os.path.splitext(os.path.basename(imagem))[0]

    otsu = segmentacao_otsu(imagem)
    kmeans = segmentacao_kmeans(imagem, 3)

    salvar_resultado(otsu, f"otsu_{nome_base}.jpg")
    salvar_resultado(kmeans, f"kmeans_{nome_base}.jpg")

    # Exibe resultados
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(imagem)[:, :, ::-1])
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(otsu, cmap='gray')
    plt.title("Otsu")

    plt.subplot(1, 3, 3)
    plt.imshow(kmeans[:, :, ::-1])
    plt.title("K-means")

    plt.tight_layout()
    plt.show()
