import os
import sys
import numpy as np

import tkinter as tk
import _pickle as pickle

root_dir = os.path.dirname(__file__)
sys.path.append(root_dir)
from game_board import GameBoard
from pieces import Piece, Pawn, Rook, Knight, Bishop, Queen, King

def initialize_board():
    board_frame = np.empty((8, 8), dtype=object)
    column_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for row in range(8):
        for column, column_char in zip(range(8), column_chars):
            field_name = '{}{}'.format(column_char, row + 1)
            if list(filter(lambda p: p.current_field == field_name, Piece)):
                p = list(filter(lambda p: p.current_field == field_name, Piece))[0]
                board_frame[row, column] = p
            else:
                pass
    return board_frame


im_path = os.path.join(root_dir, 'images')
rookb_im = os.path.join(im_path, 'bR.png')
rookw_im = os.path.join(im_path, 'wR.png')
pawnb_im = os.path.join(im_path, 'bP.png')
pawnw_im = os.path.join(im_path, 'wP.png')
knightb_im = os.path.join(im_path, 'bN.png')
knightw_im = os.path.join(im_path, 'wN.png')
bishopb_im = os.path.join(im_path, 'bB.png')
bishopw_im = os.path.join(im_path, 'wB.png')
kingb_im = os.path.join(im_path, 'bK.png')
kingw_im = os.path.join(im_path, 'wK.png')
queenb_im = os.path.join(im_path, 'bQ.png')
queenw_im = os.path.join(im_path, 'wQ.png')

root = tk.Tk()

P_W1 = Pawn('Pawn White 1', 'P_W1', 1, 'A2', pawnw_im)
P_W2 = Pawn('Pawn White 2', 'P_W2', 1, 'B2', pawnw_im)
P_W3 = Pawn('Pawn White 3', 'P_W3', 1, 'C2', pawnw_im)
P_W4 = Pawn('Pawn White 4', 'P_W4', 1, 'D2', pawnw_im)
P_W5 = Pawn('Pawn White 5', 'P_W5', 1, 'E2', pawnw_im)
P_W6 = Pawn('Pawn White 6', 'P_W6', 1, 'F2', pawnw_im)
P_W7 = Pawn('Pawn White 7', 'P_W7', 1, 'G2', pawnw_im)
P_W8 = Pawn('Pawn White 8', 'P_W8', 1, 'H2', pawnw_im)

R_W1 = Rook('Rook White 1', 'R_W1', 1, 'A1', rookw_im)
R_W2 = Rook('Rook White 2', 'R_W2', 1, 'H1', rookw_im)

N_W1 = Knight('Knight White 1', 'N_W1', 1, 'B1', knightw_im)
N_W2 = Knight('Knight White 2', 'N_W2', 1, 'G1', knightw_im)

B_W1 = Bishop('Bishop White 1', 'B_W1', 1, 'C1', bishopw_im)
B_W2 = Bishop('Bishop White 2', 'B_W2', 1, 'F1', bishopw_im)

K_W = King('King White', 'K_W', 1, 'E1', kingw_im)

Q_W = Queen('Queen White', 'Q_W', 1, 'D1', queenw_im)


P_B1 = Pawn('Pawn Black 1', 'P_B1', 2, 'A7', pawnb_im)
P_B2 = Pawn('Pawn Black 2', 'P_B2', 2, 'B7', pawnb_im)
P_B3 = Pawn('Pawn Black 3', 'P_B3', 2, 'C7', pawnb_im)
P_B4 = Pawn('Pawn Black 4', 'P_B4', 2, 'D7', pawnb_im)
P_B5 = Pawn('Pawn Black 5', 'P_B5', 2, 'E7', pawnb_im)
P_B6 = Pawn('Pawn Black 6', 'P_B6', 2, 'F7', pawnb_im)
P_B7 = Pawn('Pawn Black 7', 'P_B7', 2, 'G7', pawnb_im)
P_B8 = Pawn('Pawn Black 8', 'P_B8', 2, 'H7', pawnb_im)

R_B1 = Rook('Rook Black 1', 'R_B1', 2, 'A8', rookb_im)
R_B2 = Rook('Rook Black 2', 'R_B2', 2, 'H8', rookb_im)

N_B1 = Knight('Knight Black 1', 'N_B1', 2, 'B8', knightb_im)
N_B2 = Knight('Knight Black 2', 'N_B2', 2, 'G8', knightb_im)

B_B1 = Bishop('Bishop Black 1', 'B_B1', 2, 'C8', bishopb_im)
B_B2 = Bishop('Bishop Black 2', 'B_B2', 2, 'F8', bishopb_im)

K_B = King('King Black', 'K_B', 2, 'E8', kingb_im)

Q_B = Queen('Queen Black', 'Q_B', 2, 'D8', queenb_im)

save_file = os.path.join(root_dir, 'saved_state.pkl')

board = initialize_board()
Piece.save_state(initializer=True)
gameboard = GameBoard(root)
gameboard.pack(side='top', fill='both', expand='False', padx=0, pady=0)
_images = dict()
for p in Piece:
    _images[p.short_name] = tk.PhotoImage(file=p.img_file)
    gameboard.addpiece(p.short_name, _images[p.short_name], p.field_idx)
root.mainloop()

os.remove(save_file)