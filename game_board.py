import numpy as np
import tkinter as tk

from chess_core import GameOps


class GameBoard(tk.Frame):
    column_chars, row_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], ['1', '2', '3', '4', '5', '6', '7', '8']
    label_column, label_row = [], []
    piece_images = {}
    piece_image_paths = {}

    def __init__(self, parent, board_state, n_rows=8, n_cols=8, field_w=64, color1='#F0D9B5', color2='#B58863'):
        """field_w is the width of a field, in pixels"""
        parent.resizable(True, True)
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.board_state = board_state
        self._saved_states = 0
        self.final_move = 0
        self.redo_move = 0
        self.state_change = False
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.color1 = color1
        self.color2 = color2
        self.pieces = {}
        self.piece = None
        self.click_idx = 0
        self.valid_move = False
        self.kill = None
        self.field_names = np.array([['{}{}'.format(i, j) for i in self.column_chars] for j in range(1, 8 + 1)])

        # get board game dimensions
        self.layout = self.get_game_layout(window_w=960, window_h=540, n_rows=n_rows, n_cols=n_cols)

        # create main window
        self.mainWindow = parent
        self.mainWindow.title('Chess')
        self.mainWindow.geometry('{}x{}'.format(self.layout['total_w'], self.layout['board_ext_h']))
        self.mainWindow.bind('<Configure>', self.refresh)

        # create board canvas
        self.board = tk.Canvas(self, borderwidth=0, highlightthickness=2,
                                width=self.layout['board_ext_w'],
                                height=self.layout['board_ext_h'],
                                background='bisque')
        self.board.bind('<Button-1>', self.click_board)  # left mouse interaction with board
        self.board.pack(side=tk.LEFT, fill=tk.Y, expand=tk.FALSE, padx=0, pady=0)
        self.board.bind('<Configure>', self.draw_board)
        self.highlighter = None  # the highlighter for game piece selection

        # create panel canvas (for additional widgets)
        self.panel = tk.Canvas(self, borderwidth=0, highlightthickness=2,
                                width=self.layout['panel_w'] + self.layout['label_w'],
                                height=self.layout['board_ext_h'],
                                background='bisque')
        self.panel.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.FALSE, padx=0, pady=0)
        self.panel.bind('<Configure>', self.draw_panel)

        # pack the entire GUI
        self.pack(side='top', fill='both', expand='False', padx=0, pady=0)
        return

    def refresh(self, event):
        self.layout = self.get_game_layout(
            window_w=self.mainWindow.winfo_width(), window_h=self.mainWindow.winfo_height(), n_rows=self.n_rows,
            n_cols=self.n_cols, last_layout=self.layout)
        return

    def draw_board(self, event):
        """Redraw the board, possibly in response to window being resized"""
        self.board.delete('square')
        self.board.config(width=self.layout['board_ext_w'], height=self.layout['board_ext_h'])
        # create row and column labels
        for row, col in zip(range(self.n_rows), range(self.n_rows)):
            self.label_row.append(tk.Label(self.board, text=self.row_chars[::-1][row], fg='black', bg='bisque'))
            self.label_column.append(tk.Label(self.board, text=self.column_chars[col], fg='black', bg='bisque'))
            self.board.create_window(
                self.layout['label_w'] / 2,
                self.layout['label_w'] + self.layout['field_w'] * row,
                window=self.label_row[row]
            )
            self.board.create_window(
                (self.layout['field_w'] * col) + 2 * self.layout['label_w'],
                self.layout['label_w'] / 2 + self.n_cols * self.layout['field_w'],
                window=self.label_column[col]
            )

        # create field squares
        color = self.color2
        for row in range(self.n_rows):
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.n_cols):
                x1 = (col * self.layout['field_w'] + self.layout['label_w'])
                y1 = (row * self.layout['field_w'])
                x2 = x1 + self.layout['field_w']
                y2 = y1 + self.layout['field_w']
                self.board.create_rectangle(x1, y1, x2, y2, outline='black', fill=color, tags='square')
                color = self.color1 if color == self.color2 else self.color2

        # place all pieces
        for name in self.pieces:
            self.placepiece(name, (abs(self.pieces[name][0]-7), self.pieces[name][1]))
        self.board.tag_raise('piece')
        self.board.tag_lower('square')

        if self.highlighter is not None:
            self.board.delete(self.highlighter)
            self.highlighter = self.create_highlighter(self.board, self.layout['click_coords'])


        return

    def draw_panel(self, event):
        """Redraw the board, possibly in response to window being resized"""
        # undo/redo buttons
        self.panel.delete('button')
        self.panel.config(width=self.layout['panel_w'] + self.layout['label_w'], height=self.layout['board_ext_h'])
        undo_x = 0.25 * self.layout['panel_w']
        undo_y = self.layout['label_w']
        redo_x = undo_x
        redo_y = undo_y + self.layout['label_w']

        b_undo = self.make_button(
            button_name='UNDO', button_method=self.clickUndo, canvas=self.panel, x=undo_x, y=undo_y, anchor='w')
        b_redo = self.make_button(
            button_name='REDO', button_method=self.clickRedo, canvas=self.panel, x=undo_x, y=redo_y, anchor='w')
        # self.board.create_rectangle(
        #     undo_x - 2 * self.layout['label_w'],
        #     undo_y + 2 * self.layout['label_w'],
        #     redo_x + 2 * self.layout['label_w'],
        #     redo_y - 2 * self.layout['label_w']
        # )
        return

    @staticmethod
    def get_game_layout(window_w, window_h, n_rows, n_cols, last_layout=None):
        """creates the dimensions for a GUI board game including field labels and a user interface panel:
        Arguments:
            window_w: GUI window width
            window_h: GUI window height
            n_rows: the number of field rows (for chess: n_rows=8)
            n_cols: the number of field columns (for chess: n_cols=8)
        Returns:
            layout: (dict) a dict containing all relevant variable for the layout
        """
        layout = {}
        window_square_length = min(window_w, window_h)
        field_w = int(window_square_length / (n_cols + 0.5))
        layout['scale'] = field_w / last_layout['field_w'] if last_layout is not None else 1
        layout['field_w'] = field_w
        layout['label_w'] = int(layout['field_w'] / 2)
        layout['board_x'] = layout['label_w']
        layout['board_w'] = int(n_cols * layout['field_w'])
        layout['board_h'] = layout['board_w']
        layout['board_ext_h'] = int(window_square_length)
        layout['board_ext_w'] = int(layout['board_w'] + layout['label_w'])
        layout['panel_x'] = int(layout['board_ext_w'] + layout['label_w'])
        layout['panel_w'] = int(window_w - layout['panel_x'])
        layout['total_w'] = int(layout['panel_x'] + layout['panel_w'])
        if last_layout is None:
            layout['click_coords'] = (0, 0)
        else:
            x, y = last_layout['click_coords']
            layout['click_coords'] = (x * layout['scale'], y * layout['scale'])
        return layout

    @staticmethod
    def make_button(button_name, button_method, canvas, x, y, anchor='c'):
        """creates a button with a name and method
        Arguments:
            button_name: (str) the name of the button
            button_method: (method) the method to execute upon button click
            canvas: (tkinter.Tk.Canvas obj) the tkinter canvas obj in which to place the button
            x, y: (int) button placement coordinates
            anchor: (str) button placement anchor ('n', 'e', 's', 'w', 'c', ...)
        Returns:
            button: tkinter button widget
        """
        button = tk.Button(canvas, text=button_name)
        canvas.create_window(x, y, window=button, tags='button', anchor=anchor)
        button.bind('<Button-1>', button_method)
        return button

    def click_board(self, event):
        """This function controls the game board interaction
            1) the first click (click_idx==0) selects the chess piece
            2) the second click will move the piece to the desired field (if move is valid)
            3) in case of an invalid move, the user will be informed
        """
        if GameOps.is_checkmate:
            return
        field_idx = self.coord2field(event)
        if field_idx is None:
            return
        self.layout['click_coords'] = (event.x, event.y)
        piece = self.board_state[field_idx]
        field_name = self.field_names[field_idx]
        if self.click_idx == 0:
            if piece:
                self.highlighter = self.create_highlighter(canvas=self.board, click_coords=self.layout['click_coords'])
                print('clicked on {} on field {}'.format(piece.name, field_idx))
                self.piece = piece
                self.click_idx += 1
            else:
                self.piece = None
                print('no piece on field {}'.format(field_idx))
        else:
            self.piece.move(field_name)
            if self.valid_move:
                if self.kill is not None:
                    self.piece_images[self.kill.short_name] = ''
                self.placepiece(self.piece.short_name, field_idx)
                if GameOps.is_rochade_gui:
                    rochade_field_idx = GameOps.rochade_rook.get_field_idx(GameOps.rochade_move_to)
                    rochade_field_idx = (rochade_field_idx[0][0], rochade_field_idx[1][0])
                    self.placepiece(GameOps.rochade_rook.short_name, rochade_field_idx)
                    GameOps.is_rochade_gui = False
                    GameOps.is_rochade = False
                self.valid_move = False
                self.kill = None
                self.final_move = GameOps.move_count - 1
                self.redo_move = self.final_move
                GameOps.save_state()
            self.click_idx = 0
            self.board.delete(self.highlighter)
        return

    def coord2field(self, event):
        """converts the event coordinates to the corresponding game board field index
        Arguments:
            event: left mouse click event on game board
        Returns:
            field_idx: clicked game board field index
        """
        x = event.x - self.layout['board_x']
        y = event.y
        if (self.layout['board_w'] > x > 0) and (self.layout['board_h'] > y > 0):
            field_idx = (abs(int(y / self.layout['field_w']) - 7), int(x / self.layout['field_w']))
        else:
            field_idx = None
        return field_idx

    def create_highlighter(self, canvas, click_coords):
        return canvas.create_rectangle(self.rectangle_coords(click_coords), width=4, tags='square')

    def rectangle_coords(self, coords):
        x, y = coords
        x_0 = int((x - self.layout['label_w']) / self.layout['field_w']) * self.layout['field_w'] + self.layout['label_w']
        y_0 = int(y / self.layout['field_w'] + 1) * self.layout['field_w']
        x_1 = x_0 + self.layout['field_w']
        y_1 = y_0 - self.layout['field_w']
        return x_0, y_0, x_1, y_1

    def addpiece(self, name, image, field_idx):
        """Add a piece to the playing board"""
        self.board.create_image(0, 0, image=image, tags=(name, 'piece'), anchor='c')
        self.placepiece(name, (field_idx[0][0], field_idx[1][0]))

    @classmethod
    def addimages(cls, piece_images, piece_image_paths):
        cls.piece_images = piece_images
        cls.piece_image_paths = piece_image_paths
        return

    def placepiece(self, name, field_idx):
        """Place a piece at the given row/column"""
        row = abs(field_idx[0] - 7)
        column = field_idx[1]
        self.pieces[name] = (row, column)
        x0 = self.layout['field_w'] * (0.5 + column) + self.layout['label_w']
        y0 = self.layout['field_w'] * (0.5 + row)
        self.board.coords(name, x0, y0)

    def loadStates(self, state_count):
        if not self.final_move >= state_count >= -1:
            print('cannot progress further')
            return
        if self.redo_move == self.final_move:
            self.state_change = True
            self._saved_states = GameOps.load_state()
        self.redo_move = state_count
        state = self._saved_states[self.redo_move].copy()
        print('Loaded Move {}'.format('Initial' if state_count == -1 else state_count))
        GameOps.pieces = [p for p in state['pieces'] if p.is_alive]
        GameOps.move_count = state['globalVars']['move_count']
        GameOps.was_checked = state['globalVars']['was_checked']
        GameOps.is_rochade, GameOps.is_rochade_gui = state['globalVars']['is_rochade'],\
                                                 state['globalVars']['is_rochade_gui']
        GameOps.rochade_rook = state['globalVars']['rochade_rook']
        GameOps.rochade_move_to = state['globalVars']['rochade_move_to']
        GameOps.rochade_field_idx = state['globalVars']['rochade_field_idx']
        GameOps.queen_counter = state['globalVars']['queen_counter']
        GameOps.is_checkmate = state['globalVars']['is_checkmate']
        for i in range(len(self.board_state)):
            for j in range(len(self.board_state[i])):
                self.board_state[i][j] = None
                self.board_state[i][j] = state['board'][i][j]
        self.board.delete('piece')
        for p in GameOps:
            self.piece_images[p.short_name] = tk.PhotoImage(file=p.img_file)
            self.addpiece(p.short_name, self.piece_images[p.short_name], p.field_idx)
        return

    def clickUndo(self, event):
        state_count = self.redo_move - 1
        self.loadStates(state_count)
        return

    def clickRedo(self, event):
        state_count = self.redo_move + 1
        self.loadStates(state_count)
        return
