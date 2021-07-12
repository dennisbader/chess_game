import os
import numpy as np
import pandas as pd
import tkinter as tk

from chess_core import Pawn, Rook, Knight, Bishop, Queen, King


class ChessConfig:
    @staticmethod
    def instantiate_pieces(piece_specs_fin, img_path):
        piece_specs = pd.read_csv(piece_specs_fin)
        for idx, piece in piece_specs.iterrows():
            piece_type = piece['name_long'].split()[0]
            piece_img_fin = os.path.join(img_path, piece['img_file'])
            specs = piece[['name_long', 'name_short', 'color', 'field']].tolist() + [piece_img_fin]
            if piece_type == 'Pawn':
                Pawn(*specs)
            elif piece_type == 'Rook':
                Rook(*specs)
            elif piece_type == 'Knight':
                Knight(*specs)
            elif piece_type == 'Bishop':
                Bishop(*specs)
            elif piece_type == 'King':
                King(*specs)
            elif piece_type == 'Queen':
                Queen(*specs)
        return

    @staticmethod
    def initialize_board(pieces):
        board_frame = np.empty((8, 8), dtype=object)
        for piece in pieces:
            board_frame[piece.field_idx] = piece
        return board_frame

    @staticmethod
    def fill_board(pieces, board_gui):
        piece_images = {}
        piece_image_paths = {}
        for piece in pieces:
            piece_images[piece.short_name] = tk.PhotoImage(file=piece.img_file)
            piece_image_paths[piece.short_name[:3]] = piece.img_file
            board_gui.add_piece(piece.short_name, piece_images[piece.short_name], piece.field_idx)
        board_gui.add_images(piece_images, piece_image_paths)
        return
