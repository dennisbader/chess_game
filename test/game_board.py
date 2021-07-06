import os
import numpy as np

import tkinter as tk
import _pickle as pickle


class GameBoard(tk.Frame):
    column_chars, row_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], ['1', '2', '3', '4', '5', '6', '7', '8']
    label_column, label_row = [], []

    def __init__(self, parent, n_rows=8, n_cols=8, field_w=64, color1='#F0D9B5', color2='#B58863'):
        """field_w is the width of a field, in pixels"""
        self._saved_states = 0
        self.final_move = 0
        self.redo_move = 0
        self.state_change = False
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.field_w = field_w
        self.color1 = color1
        self.color2 = color2
        self.pieces = {}
        self.piece = 0
        self.click_idx = 0
        self.valid_move = False
        self.kill = False
        self.field_names = np.array([['{}{}'.format(i, j) for i in self.column_chars] for j in range(1, 8 + 1)])

        # get board game dimensions
        self.board_w, self.board_h, self.label_w, self.board_ext_w, self.board_ext_h, self.panel_w \
            = self.get_gui_dimensions(field_w=field_w, n_rows=n_rows, n_cols=n_cols)

        parent.resizable(False, False)
        tk.Frame.__init__(self, parent)
        self.mainWindow = parent
        self.mainWindow.title('Chess')
        self.mainWindow.geometry('{}x{}'.format(
            int(self.board_w + self.label_w + self.panel_w), self.board_h))

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=self.board_w + self.label_w,
                                height=self.board_h + self.label_w,
                                background='bisque')
        self.canvas.bind('<Button-1>', self.callback)
        # undo/redo buttons
        undo_x = self.board_w + self.label_w + 0.25 * self.panel_w
        redo_x = undo_x + 0.5 * self.panel_w
        buttons_y0 = self.field_w / 2

        self.make_button(button_name='UNDO', button_method=self.clickUndo, canvas=self.canvas, x=undo_x, y=buttons_y0)
        self.make_button(button_name='REDO', button_method=self.clickRedo, canvas=self.canvas, x=redo_x, y=buttons_y0)

        self.canvas.pack(side='top', fill=tk.BOTH, expand=tk.FALSE, padx=0, pady=0)

        # this binding will cause a refresh if the user interactively
        # changes the window size
        self.canvas.bind('<Configure>', self.refresh)
        return

    @staticmethod
    def get_gui_dimensions(field_w, n_rows, n_cols):
        """creates the dimensions for a GUI board game including field labels and a user interface panel:
        Arguments:
            field_w: the width of the square chess fields
            n_rows: the number of field rows (for chess: n_rows=8)
            n_cols: the number of field columns (for chess: n_cols=8)
        Returns:
            board_(w/h): the (width/height) of the board including only playable fields
            label_w: the column/row label width
            board_ext_(w/h): the (width/height) of the board including playable fields and labels
            panel_w: the width of the panel user interface (for additional features to the game)
        """

        board_w = n_cols * field_w
        board_h = n_rows * field_w
        label_w = field_w / 2
        board_ext_w = board_w + 2 * label_w
        board_ext_h = board_h + label_w
        panel_w = board_w / 2
        return board_w, board_h, label_w, board_ext_w, board_ext_h, panel_w

    @staticmethod
    def make_button(button_name, button_method, canvas, x, y):
        """creates a button with a name and method"""
        button = tk.Button(canvas, text=button_name)
        canvas.create_window(x, y, window=button, tags='button')
        button.bind('<Button-1>', button_method)
        return button

    def coord2field(self, event):
        if self.board_h + self.label_w > event.x > self.label_w and event.y <= self.board_h:
            x = event.x - self.label_w
            y = event.y
            field_idx = (abs(int(y / self.field_w) -7), int(x / self.field_w))
        else:
            field_idx = False
        return field_idx

    def callback(self, event):
        if Piece.is_checkmate:
            return
        field_idx = self.coord2field(event)
        if field_idx:
            piece = board[field_idx]
            field_name = self.field_names[field_idx]
            if self.click_idx > 0:
                self.piece.move(field_name)
                if self.valid_move:
                    if self.kill:
                        _images[self.kill.short_name] = ''
                    self.placepiece(self.piece.short_name, field_idx)
                    if Piece.is_rochade_gui:
                        rochade_field_idx = Piece.rochade_rook.get_field_idx(Piece.rochade_move_to)
                        rochade_field_idx = (rochade_field_idx[0][0], rochade_field_idx[1][0])
                        self.placepiece(Piece.rochade_rook.short_name, rochade_field_idx)
                        Piece.is_rochade_gui = False
                        Piece.is_rochade = False
                    self.valid_move = False
                    self.kill = False
                    self.final_move = Piece.move_count - 1
                    self.redo_move = self.final_move
                    Piece.save_state()
                self.click_idx = 0
                self.canvas.delete(self.highlighter)
            else:
                if piece:
                    self.highlighter = self.canvas.create_rectangle(self.rectangle_coords(event, field_idx), width=4)
                    print('clicked on {} on field {}'.format(piece.name, field_idx))
                    self.piece = piece
                    self.click_idx += 1
                else:
                    self.piece = 0
                    print('no piece on field ', self.coord2field(event))
        return

    def rectangle_coords(self, event, field_idx):
        x_0 = int((event.x - self.label_w) / self.field_w) * self.field_w + self.label_w
        y_0 = int(event.y / self.field_w + 1) * self.field_w
        x_1 = x_0 + self.field_w
        y_1 = y_0 - self.field_w
        return x_0, y_0, x_1, y_1

    def addpiece(self, name, image, field_idx):
        """Add a piece to the playing board"""
        self.canvas.create_image(0,0, image=image, tags=(name, 'piece'), anchor='c')
        self.placepiece(name, (field_idx[0][0], field_idx[1][0]))

    def placepiece(self, name, field_idx):
        """Place a piece at the given row/column"""
        row = abs(field_idx[0] - 7)
        column = field_idx[1]
        self.pieces[name] = (row, column)
        x0 = (column * self.field_w) + int(self.field_w/2) + self.label_w
        y0 = (row * self.field_w) + int(self.field_w/2)
        self.canvas.coords(name, x0, y0)

    def loadStates(self, state_count):
        if not self.final_move >= state_count >= -1:
            print('cannot progress further')
            return
        if self.redo_move == self.final_move:
            self.state_change = True
            self._saved_states = Piece.load_state()
        self.redo_move = state_count
        state = self._saved_states[self.redo_move].copy()
        print('Loaded Move {}'.format('Initial' if state_count == -1 else state_count))
        Piece._registry = [p for p in state['pieces'] if p.is_alive]
        Piece.move_count = state['globalVars']['move_count']
        Piece.was_checked = state['globalVars']['was_checked']
        Piece.is_rochade, Piece.is_rochade_gui = state['globalVars']['is_rochade'],\
                                                 state['globalVars']['is_rochade_gui']
        Piece.rochade_rook = state['globalVars']['rochade_rook']
        Piece.rochade_move_to = state['globalVars']['rochade_move_to']
        Piece.rochade_field_idx = state['globalVars']['rochade_field_idx']
        Piece.queen_counter = state['globalVars']['queen_counter']
        Piece.is_checkmate = state['globalVars']['is_checkmate']
        for i in range(len(board)):
            for j in range(len(board[i])):
                board[i][j] = None
                board[i][j] = state['board'][i][j]
        self.canvas.delete('piece')
        for p in Piece:
            _images[p.short_name] = tk.PhotoImage(file=p.img_file)
            self.addpiece(p.short_name, _images[p.short_name], p.field_idx)
        return

    def clickUndo(self, event):
        state_count = self.redo_move - 1
        self.loadStates(state_count)
        return

    def clickRedo(self, event):
        state_count = self.redo_move + 1
        self.loadStates(state_count)
        return

    def refresh(self, event):
        """Redraw the board, possibly in response to window being resized"""
        xsize = int((event.width - (self.label_w + 2)) / self.n_cols)
        ysize = int((event.height - (self.label_w + 2)) / self.n_rows)
        self.field_w = min(xsize, ysize)
        self.label_w = self.field_w / 2
        self.board_h = self.n_rows * self.field_w
        self.board_w = self.n_cols * self.field_w
        self.panel_w = self.board_w / 2
        self.canvas.delete('square')
        color = self.color2
        for row in range(self.n_rows):
            self.label_column.append(tk.Label(self.canvas, text=self.column_chars[-1 - row], fg='black', bg='bisque'))
            self.label_row.append(tk.Label(self.canvas, text=self.row_chars[row], fg='black', bg='bisque'))
            self.canvas.create_window(self.label_w / 2, self.label_w + self.field_w * row,
                                      window=self.label_column[row])
            self.canvas.create_window((self.field_w * row) + 2 * self.label_w, self.label_w / 2 + self.n_rows * self.field_w,
                                      window=self.label_row[row])
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.n_cols):
                x1 = (col * self.field_w + self.label_w)
                y1 = (row * self.field_w)
                x2 = x1 + self.field_w
                y2 = y1 + self.field_w
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='black', fill=color, tags='square')
                color = self.color1 if color == self.color2 else self.color2
        # placement of pieces
        for name in self.pieces:
            self.placepiece(name, (abs(self.pieces[name][0]-7), self.pieces[name][1]))
        self.canvas.tag_raise('piece')
        self.canvas.tag_lower('square')
        return

