import numpy as np

FloydSteinberg = np.array([
    [0, 0, 0],
    [0, 0, 7],
    [3, 5, 1]
])

FalseFloydSteinberg = np.array([
    [0, 0, 0],
    [0, 0, 3],
    [0, 3, 2]
])

Jarvis = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1]
])

Stucki = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2],
    [1, 2, 4, 2, 1]
])

Atkinson = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
])

Burkes = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2],
    [0, 0, 0, 0, 0]
])

Sierra32 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 5, 3],
    [2, 4, 5, 4, 2],
    [0, 2, 3, 2, 0]
])

Sierra16 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 4, 3],
    [1, 2, 3, 2, 1],
    [0, 0, 0, 0, 0]
])

Sierra4 = np.array([
    [0, 0, 0],
    [0, 0, 2],
    [1, 1, 0]
])

all_masks = {
    'floyd-steinberg': FloydSteinberg,
    'false-floyd-steinberg': FalseFloydSteinberg,
    'jarvis': Jarvis,
    'stucki': Stucki,
    'atkinson': Atkinson,
    'burkes': Burkes,
    'sierra32': Sierra32,
    'sierra16': Sierra16,
    'sierra4': Sierra4
}
