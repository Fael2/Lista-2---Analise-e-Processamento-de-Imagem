import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tkinter as tk
from tkinter import ttk, messagebox

def calcular_fecho_convexo(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

    img = cv2.imread(img_path, 0)
    if img is None:
        raise FileNotFoundError("Erro ao carregar imagem. Verifique o caminho e a extensão.")

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("Nenhum contorno encontrado. Verifique se a imagem está corretamente segmentada.")

    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color, [c], -1, (0, 255, 0), 2)
    cv2.drawContours(img_color, [hull], -1, (0, 0, 255), 2)

    return img, img_color


def salvar_resultado(img, nome_arquivo):
    pasta_resultados = "../result_Convex"
    os.makedirs(pasta_resultados, exist_ok=True)
    caminho = os.path.join(pasta_resultados, nome_arquivo)
    cv2.imwrite(caminho, img)
    print(f"✅ Imagem salva em: {caminho}")

def selecionar_imagem_modal(pasta="../results"):
    imagens = [
        f for f in os.listdir(pasta)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        and not f.lower().startswith("kmeans_")
        and not f.lower().startswith("convex_")
    ]

    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem válida encontrada na pasta {pasta}")

    imagem_selecionada = {"nome": None}

    def confirmar():
        selecao = combo.get()
        if not selecao:
            messagebox.showwarning("Atenção", "Selecione uma imagem antes de continuar.")
        else:
            imagem_selecionada["nome"] = selecao
            root.destroy()

    root = tk.Tk()
    root.title("Selecione uma imagem da pasta 'results'")
    root.geometry("400x200")
    root.resizable(False, False)

    label = ttk.Label(root, text="Escolha a imagem segmentada (excluindo K-means):")
    label.pack(pady=10)

    combo = ttk.Combobox(root, values=imagens, state="readonly", width=50)
    combo.pack(pady=5)
    combo.current(0)

    botao_confirmar = ttk.Button(root, text="Confirmar", command=confirmar)
    botao_confirmar.pack(pady=20)

    root.mainloop()

    if not imagem_selecionada["nome"]:
        raise ValueError("Nenhuma imagem foi selecionada.")
    return os.path.join(pasta, imagem_selecionada["nome"])


if __name__ == "__main__":
    try:
        imagem = selecionar_imagem_modal()
        nome_base = os.path.splitext(os.path.basename(imagem))[0].replace("otsu_", "")

        print(f"🔍 Calculando fecho convexo da imagem: {imagem}")

        segmentada, convexa = calcular_fecho_convexo(imagem)
        salvar_resultado(convexa, f"convex_{nome_base}.jpg")

        # Exibe resultados
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(segmentada, cmap='gray')
        plt.title("Imagem Segmentada (Otsu)")

        plt.subplot(1, 2, 2)
        plt.imshow(convexa[:, :, ::-1])
        plt.title("Fecho Convexo (vermelho)")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Erro: {e}")
