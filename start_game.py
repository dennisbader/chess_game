import os
import tkinter as tk

from game_config import ChessConfig
from game_board import GameBoard
from chess_core import GameOps


root_dir = os.path.dirname(__file__)
img_path = os.path.join(root_dir, 'images')
piece_specs_fin = os.path.join(root_dir, 'piece_specs.csv')
save_file = os.path.join(root_dir, 'saved_state.pkl')

ChessConfig.instantiate_pieces(piece_specs_fin, img_path)
board_objects = ChessConfig.initialize_board(GameOps.pieces)

root = tk.Tk()
board_gui = GameBoard(root, board_objects, root_dir)
GameOps.initialize_game(board_objects=board_objects, board_gui=board_gui)
GameOps.verbose = True
ChessConfig.fill_board(pieces=GameOps.pieces, board_gui=board_gui)
root.mainloop()
os.remove(save_file)