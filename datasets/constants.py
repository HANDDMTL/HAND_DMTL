
import numpy as np

NUM_KPS = 21

BONE_HIERARCHY = np.array([
    [0, 0],

    [1,0], 
    [2,1], 
    [3,2],
    [4,3],

    [5,0],
    [6,5],
    [7,6],
    [8,7],

    [9,0],
    [10,9],
    [11,10],
    [12,11],

    [13,0],
    [14,13],
    [15,14],
    [16,15],

    [17,0],
    [18,17],
    [19,18],
    [20,19],
])

BONE_COLORS = np.array([
    [0.4, 0.4, 0.4],
    [0.4, 0.0, 0.0],
    [0.6, 0.0, 0.0],
    [0.8, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.4, 0.4, 0.0],
    [0.6, 0.6, 0.0],
    [0.8, 0.8, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.4, 0.2],
    [0.0, 0.6, 0.3],
    [0.0, 0.8, 0.4],
    [0.0, 1.0, 0.5],
    [0.0, 0.2, 0.4],
    [0.0, 0.3, 0.6],
    [0.0, 0.4, 0.8],
    [0.0, 0.5, 1.0],
    [0.4, 0.0, 0.4],
    [0.6, 0.0, 0.6],
    [0.7, 0.0, 0.8],
    [1.0, 0.0, 1.0]
])*255.0
