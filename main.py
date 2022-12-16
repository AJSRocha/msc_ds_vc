import os
import matplotlib.pyplot as plt
import skimage
import numpy as np
from scipy import ndimage as ndi
from skimage import io
import pandas as pd
import scipy
import cv2

# Importa dados
dados = pd.DataFrame()
# importa nome dos ficheiros
lista = []
for file in os.listdir('photos/bios'):
    lista.append(file)
dados['file'] = lista
# extrai atributos dos ficheiros com base nos nomes
res = []
for name in dados['file']:
    res.append(name.split('-'))

dados['Lt'] = [item[5] for item in res]
dados['Lt'] = dados['Lt'].apply(lambda x: float(x.split()[0].replace(',','.')))
dados['month'] = [item[3][2:5] for item in res]
dados['semester'] = ['S1' if month in ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun'] else 'S2' for month in dados['month']]
# prepara idades
dados['age'] = 0
#limpeza
del lista, res

def suavizador(perfil:list, h=3):
    res = np.convolve(perfil, np.ones(h), "valid")/h
    return res

def deriv(perfil: list, ordem: int = 1):
    res = []
    if ordem == 1:
        for i in range(ordem, np.shape(perfil)[0] - ordem):
            res.append((perfil[i + ordem] - perfil[i - ordem]) / 2*ordem)
    else:
        for i in range(ordem, np.shape(perfil)[0] - ordem):
            res_n = 0
            for j in range(1, ordem+1):
                res_n += (perfil[i + j] - perfil[i - j])
            res.append(res_n/(2*ordem))
    return res

def conta_transicoes(trans:list):
    trans_b = []
    dist = []
    for i in range(1, np.shape(trans)[0]-1):
        if trans[i+1] > 0 and trans[i] < 0:
            trans_b.append('H-O')
            dist.append(i)
        elif trans[i+1] < 0 and trans[i] > 0:
            trans_b.append('O-H')
    trans_limpo = [item for item in trans_b if item == 'H->0']
    # bordo
    bordo = trans_b[-1]
    return trans_limpo, dist, bordo

def drop_falsos(lista:list, lista_valid):
    k = 0
    for i, j in zip(lista, lista[1:]):
        if i < j:
            lista_valid[k] = 0
        k += 1
    return lista_valid

def drop_falsos_reverse(lista:list, lista_valid):
    k = 0
    for i, j in zip(lista, lista[1:]):
        if i < j:
            lista_valid[k+1] = 0
        k += 1
    return lista_valid

def distanciador(lista: list):
    res = [lista[indice + 1] - lista[indice] for indice in range(len(lista) - 1)]
    return res

def processador(imagem,
                plot:bool=True,
                plot_aneis:bool = False,
                semester = 'S1',
                kernel:int = 101,
                outward:bool = True,
                smooth:int = 3,
                perf_deriv:int =1,
                layer:str = "raw",
                perfil:str = 'raw'):

    if layer == 'raw':
        colapsado = skimage.color.rgb2gray(imagem)
    elif layer == 'red':
        colapsado = imagem[:,:,0]/255

    # prepara histograma para desenhar imagem
    histogram, bin_edges = np.histogram(colapsado, bins=100, range=(0, 1))

    # detecta a borda e cria uma mascara
    markers = np.zeros_like(colapsado)
    markers[colapsado < 0.1] = 1 # como determinar este valor?
    markers[colapsado> 0.8] = 2 # como determinar este valor?

    edge = skimage.filters.sobel(colapsado)
    segmentation_edge = skimage.segmentation.watershed(edge, markers)

    # identifica o objecto maior
    otolito = skimage.filters.apply_hysteresis_threshold(colapsado*segmentation_edge, high = 0.6, low = 0.4) # validar valores
    labeled = ndi.label(otolito)[0]


    # identifica o centroide
    largest_nonzero_label = np.argmax(np.bincount(labeled[labeled > 0]))
    binary = labeled == largest_nonzero_label
    skeleton = skimage.morphology.skeletonize(binary)
    g, nodes = skimage.graph.pixel_graph(skeleton, connectivity=2)
    px, distances = skimage.graph.central_pixel(
    g, nodes=nodes, shape=skeleton.shape, partition_size=100)

    centroid = skimage.measure.centroid(labeled > 0)

    # identifica a borda pixel a pixel
    borda = skimage.measure.find_contours(binary) # em vez de segmentation edge?

    # calcula extremo do rostro
    origin_x, origin_y = px[1], px[0]
    res = []
    for array in borda:
        for line in array:
            res.append(line.tolist())

    distancia = []
    for item in res:
        distancia.append(np.sqrt((origin_y-item[1])**2 + (origin_x-item[0])**2))
    max_index = distancia.index(max(distancia))
    extremo = res[max_index]
    origin = [origin_x, extremo[1]]
    dest = [origin_y,extremo[0]]

    # Versao smoothed da imagem
    hpf = colapsado - cv2.GaussianBlur(colapsado,(kernel,kernel),0)

    # linha de perfil a partir do centroide - pode ser feita a partir do colapsado ou do hpf
    profile = skimage.measure.profile_line(colapsado, src = (origin_y, origin_x), dst = (extremo[0],extremo[1]))
    profile_hpf = skimage.measure.profile_line(hpf, src = (origin_y, origin_x), dst = (extremo[0],extremo[1]))

    # histograma do perfil; intuiçao para smoothing
    p_hist, p_bin = np.histogram(profile, bins=255, range=(0, 1))

    #contagem de aneis
    if perfil == 'raw':
        smoothed = suavizador(profile, smooth)
    else:
        smoothed = suavizador(profile_hpf, smooth)

    smoothed_deriv = deriv(smoothed, ordem= perf_deriv)

    _,transicoes,bordo = conta_transicoes(smoothed_deriv)

    # começa aqui a nova funcao
    valid = np.ones_like(transicoes)
    transicoes_validas = transicoes.copy()
    print(transicoes_validas)

    contador = 0
    while contador <= len(transicoes):
        diff = distanciador(transicoes_validas)
        #print(f'diff = {diff}')
        if outward:
            valid = drop_falsos(diff, valid)
        else:
            valid = drop_falsos_reverse(diff, valid)
        transicoes_validas = [transicoes_validas[i] * valid[i] for i in range(len(transicoes_validas))]
        transicoes_validas = [i for i in transicoes_validas if i != 0]
        valid = np.ones_like(transicoes_validas)
        #print(transicoes_validas)
        contador += 1

    if bordo == 'O-H' and semester == 'S1':
        bordo = 1
    else:
        bordo = 0

    age = 0
    for i in transicoes_validas:
        if i != 0:
            age +=1
    age += bordo

    # visualização de alguns picos
    picos = scipy.signal.find_peaks(smoothed, prominence = 0.012) # manipular distance e width
    vales = scipy.signal.find_peaks(smoothed*-1, prominence = 0.012)

    # plots
    if plot:
        f, ax = plt.subplots(3,3, figsize = (40,20))
        # imagem original em grayscale
        ax[0,0].imshow(colapsado, cmap = plt.cm.gray)
        # imagem original depois de retirar o fundo por watershed
        ax[0,1].imshow(colapsado * mascara, cmap = plt.cm.gray)
        # histograma da imagem original
        ax[0,2].plot(bin_edges[0:-1], histogram)
        # regiao do otolito determinada por hysteresis
        ax[1,0].imshow(otolito, cmap = plt.cm.gray)
        # centroide, borda e linha de perfil do otolito

        ax[1,1].imshow(colapsado, cmap = plt.cm.gray)
        ax[1,1].plot(origin, dest, linewidth = 5, color = 'r')
        ax[1,1].scatter(px[1], px[0], label='graph center')
        ax[1,1].scatter(centroid[1], centroid[0], label='centroid')
        ax[1,1].legend()
        ax[1,1].set_axis_off()
        ax[1,1].set_title('graph center vs centroid')
        for contour in borda:
            ax[1,1].plot(contour[:, 1], contour[:, 0], linewidth=4, color = 'r')

        # perfil desenhado do otolito
        ax[1,2].plot(profile)
        # histograma do perfil

        ax[2,1].imshow(hpf, cmap = plt.cm.gray)
        ax[2,1].plot(origin, dest, linewidth = 5, color = 'r')
        ax[2,1].scatter(px[1], px[0], label='graph center')
        ax[2,1].scatter(centroid[1], centroid[0], label='centroid')
        ax[2,1].legend()
        ax[2,1].set_axis_off()
        ax[2,1].set_title('graph center vs centroid')
        for contour in borda:
            ax[2,1].plot(contour[:, 1], contour[:, 0], linewidth=4, color = 'r')

        ax[2,2].plot(profile_hpf)
        # histograma do perfil

        ax[2,0].plot(smoothed)
        for pico in picos[0]:
            ax[2,0].vlines(x = pico, ymin = smoothed.min(), ymax= smoothed.max(), color = 'r')
        for vale in vales[0]:
            ax[2,0].vlines(x = vale, ymin = smoothed.min(), ymax= smoothed.max(), color = 'g')

    if plot_aneis:

        fig, ax = plt.subplots()
        if perfil == 'raw':
            ax.imshow(colapsado, cmap = plt.cm.gray)
        else:
            ax.imshow(hpf, cmap = plt.cm.gray)
        ax.plot(origin, dest, linewidth = 5, color = 'r')
        ax.scatter(px[1], px[0], label='graph center')

        for contour in borda:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=4, color = 'r')

        for anel in transicoes:
            dist = anel
            circulo = plt.Circle((px[1], px[0]), radius=dist, color='yellow', fill = False)
            ax.add_patch(circulo)

        for anel in transicoes_validas:
            dist = anel
            circulo = plt.Circle((px[1], px[0]), radius=dist, color='green', fill = False)
            ax.add_patch(circulo)

    print(f'idade estimada é {age} e o bordo é {bordo}')
    return age

leituras = []
for line in range(dados.shape[0]):
    idade = processador(io.imread(('photos/bios/'+dados['file'][line])), False,
                        kernel = 151, smooth=7, layer = 'red', perfil = 'raw')
    leituras.append(idade)

dados.to_csv('leituras.csv')

