import os
import numpy as np
from matplotlib import pyplot as plt

import tkinter as tk
import _pickle as pickle


class GameBoard(tk.Frame):
    column_chars, row_chars = ["A", "B", "C", "D", "E", "F", "G", "H"], ["1", "2", "3", "4", "5", "6", "7", "8"]
    label_column, label_row = [], []
    def __init__(self, parent, rows=8, columns=8, size=64, color1='#F0D9B5', color2='#B58863'):
        '''size is the size of a square, in pixels'''
        self._saved_states = 0
        self.final_move = 0
        self.redo_move = 0
        self.state_change = False
        self.rows = rows
        self.columns = columns
        self.size = size
        self.color1 = color1
        self.color2 = color2
        self.pieces = {}
        self.piece = 0
        self.click_idx = 0
        self.valid_move = False
        self.kill = False
        self.label_space = self.size / 2
        self.field_names = np.array([["{}{}".format(i, j) for i in self.column_chars] for j in range(1, 8 + 1)])
        self.canvas_width = columns * self.size
        self.canvas_height = rows * self.size
        self.board_width = self.canvas_width + 2 * self.label_space
        self.board_height = self.canvas_height + self.label_space
        self.additional_width = self.canvas_width / 2
        parent.resizable(False,False)
        tk.Frame.__init__(self, parent)
        self.mainWindow = parent
        self.mainWindow.title("Chess")
        self.mainWindow.geometry("{}x{}".format(int(self.canvas_width + self.label_space + self.additional_width) , self.canvas_height))

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=self.canvas_width + self.label_space,
                                height=self.canvas_height + self.label_space,
                                background="bisque")
        self.canvas.bind("<Button-1>", self.callback)
        # undo/redo buttons
        self.undo = tk.Button(self.canvas, text='UNDO')
        self.canvas.create_window(self.canvas_width + self.label_space + 0.25 * self.additional_width,
                                  self.size / 2, window=self.undo, tags='button')
        self.undo.bind("<Button-1>", self.clickUndo)

        self.redo = tk.Button(self.canvas, text='REDO')
        self.canvas.create_window(self.canvas_width + self.label_space + 0.75 * self.additional_width,
                                  self.size / 2, window=self.redo, tags='button')
        self.redo.bind("<Button-1>", self.clickRedo)

        self.canvas.pack(side="top", fill=tk.BOTH, expand=tk.FALSE, padx=0, pady=0)

        # this binding will cause a refresh if the user interactively
        # changes the window size
        self.canvas.bind("<Configure>", self.refresh)

    def coord2field(self, event):
        if self.canvas_height + self.label_space > event.x > self.label_space and event.y <= self.canvas_height:
            x = event.x - self.label_space
            y = event.y
            field_idx = (abs(int(y / self.size) -7), int(x / self.size))
        else:
            field_idx = False
        return field_idx

    def callback(self, event):
        # print(board[self.coord2field(event)])
        if Piece.is_checkmate:
            return
        field_idx = self.coord2field(event)
        if field_idx:
            piece = board[field_idx]
            field_name = self.field_names[field_idx]
            if self.click_idx > 0:
                # print("move to ", self.coord2field(event))
                self.piece.move(field_name)
                if self.valid_move:
                    if self.kill:
                        _images[self.kill.short_name] = ""
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
                    #Piece.load_state(Piece.move_count)
                self.click_idx = 0
                self.canvas.delete(self.highlighter)
            else:
                if piece:
                    self.highlighter = self.canvas.create_rectangle(self.rectangle_coords(event, field_idx), width=4)
                    print("clicked on {} on field {}".format(piece.name, field_idx))
                    self.piece = piece
                    self.click_idx += 1
                else:
                    self.piece = 0
                    print("no piece on field ", self.coord2field(event))
        return

    def rectangle_coords(self, event, field_idx):
        #print(event.x)
        x_0 = int((event.x - self.label_space) / self.size) * self.size + self.label_space
        y_0 = int(event.y / self.size + 1) * self.size
        x_1 = x_0 + self.size
        y_1 = y_0 - self.size
        #print(event.x, event.y)
        #print(x_0,y_0,x_1,y_1)
        return x_0, y_0, x_1, y_1

    def addpiece(self, name, image, field_idx):
        '''Add a piece to the playing board'''
        # row = abs(field_idx[0][0] - 7)
        # column = field_idx[1][0]
        self.canvas.create_image(0,0, image=image, tags=(name, "piece"), anchor="c")
        self.placepiece(name, (field_idx[0][0], field_idx[1][0]))

    def placepiece(self, name, field_idx):
        '''Place a piece at the given row/column'''
        row = abs(field_idx[0] - 7)
        column = field_idx[1]
        self.pieces[name] = (row, column)
        x0 = (column * self.size) + int(self.size/2) + self.label_space
        y0 = (row * self.size) + int(self.size/2)
        self.canvas.coords(name, x0, y0)

    def loadStates(self, state_count):
        if not self.final_move >= state_count >= -1:
            print("can't progress further")
            return
        if self.redo_move == self.final_move:
            self.state_change = True
            self._saved_states = Piece.load_state()
        self.redo_move = state_count
        state = self._saved_states[self.redo_move].copy()
        print("Loaded Move {}".format("Initial" if state_count == -1 else state_count))
        Piece._registry = [p for p in state["pieces"] if p.is_alive]
        Piece.move_count = state["globalVars"]["move_count"]
        Piece.was_checked = state["globalVars"]["was_checked"]
        Piece.is_rochade, Piece.is_rochade_gui = state["globalVars"]["is_rochade"],\
                                                 state["globalVars"]["is_rochade_gui"]
        Piece.rochade_rook = state["globalVars"]["rochade_rook"]
        Piece.rochade_move_to = state["globalVars"]["rochade_move_to"]
        Piece.rochade_field_idx = state["globalVars"]["rochade_field_idx"]
        Piece.queen_counter = state["globalVars"]["queen_counter"]
        Piece.is_checkmate = state["globalVars"]["is_checkmate"]
        for i in range(len(board)):
            for j in range(len(board[i])):
                board[i][j] = None
                board[i][j] = state["board"][i][j]
        #board = state["board"].copy()
        self.canvas.delete('piece')
        for p in Piece:
            _images[p.short_name] = tk.PhotoImage(file=p.img_file)
            self.addpiece(p.short_name, _images[p.short_name], p.field_idx)
        return


    def clickUndo(self, event):
        state_count = self.redo_move - 1
        self.loadStates(state_count)

    def clickRedo(self, event):
        state_count = self.redo_move + 1
        self.loadStates(state_count)

    def refresh(self, event):
        '''Redraw the board, possibly in response to window being resized'''
        xsize = int((event.width - (self.label_space + 2)) / self.columns)
        ysize = int((event.height - (self.label_space + 2)) / self.rows)
        self.size = min(xsize, ysize)
        self.label_space = self.size / 2
        self.canvas_height = self.rows * self.size
        self.canvas_width = self.columns * self.size
        self.additional_width = self.canvas_width / 2
        self.canvas.delete("square")
        color = self.color2
        for row in range(self.rows):
            self.label_column.append(tk.Label(self.canvas, text=self.column_chars[-1 - row], fg="black", bg="bisque"))
            self.label_row.append(tk.Label(self.canvas, text=self.row_chars[row], fg="black", bg="bisque"))
            self.canvas.create_window(self.label_space / 2, self.label_space + self.size * row,
                                      window=self.label_column[row])
            self.canvas.create_window((self.size * row) + 2 * self.label_space, self.label_space / 2 + self.rows * self.size,
                                      window=self.label_row[row])
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.columns):
                x1 = (col * self.size + self.label_space)
                y1 = (row * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=color, tags="square")
                color = self.color1 if color == self.color2 else self.color2
        # placement of pieces
        for name in self.pieces:
            self.placepiece(name, (abs(self.pieces[name][0]-7), self.pieces[name][1]))
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")

class Board:
    #@staticmethod
    def start_game():
        binary_board = np.zeros((8, 8))
        for i in range(len(board)):
            for j in range(len(board[i])):
                element = board[i][j]
                if not element:
                    binary_board[i][j] = 1
                elif element.color == 1:
                    binary_board[i][j] = 2
                else:
                    binary_board[i][j] = 0
        # plt.ion()
        # plt.pcolormesh(binary_board, cmap="gray", edgecolors='darkgrey')
        # plt.plot()
        # plt.show(block=False)
        return

class IterRegistry(type):
    def __iter__(cls):
        return iter(cls._registry)


class Piece(metaclass = IterRegistry):
    _registry = []
    column_chars = ["A", "B", "C", "D", "E", "F", "G", "H"]
    move_count = 0
    # saves if colored was already checked for rochade [0] for white, [1] for black
    was_checked = [False, False]
    is_rochade, is_rochade_gui = False, False
    rochade_rook = None
    rochade_move_to = None
    rochade_field_idx = None
    queen_counter = 0
    is_checkmate = False
    def __init__(self, name, short_name, color, current_field, img_file):
        """Init Description

        :param name: piece name
        :param color: 1 for white, 2 for black
        :param current_field: piece's starter field
        """
        self._registry.append(self)
        self.name = name
        self.short_name = short_name
        self.color = color
        self.current_field = current_field
        self.last_field = current_field
        self.field_names = np.array([["{}{}".format(i,j) for i in self.column_chars] for j in range(1,8+1)])
        self.field_idx = np.where(self.field_names == self.current_field)
        self.move_idx = 0
        self.is_alive = 1
        self.possible_moves = []
        self.checks = False
        self.img_file = img_file
        #self.img_file = tk.PhotoImage(file=img_file)
        self.transform = False

    @classmethod
    def save_state(cls, initializer=False):
        move_count = -1 if initializer else cls.move_count - 1
        if os.path.exists(save_file):
            saved_state = cls.load_state()
            os.remove(save_file)
        else:
            saved_state = dict()
        saved_state[move_count] = {"board": board.copy(),
                                        "pieces": [p for p in Piece],
                                        'globalVars': cls.get_globalVars()}

        with open(save_file, "ab") as fout:
            pickle.dump(saved_state, fout, -1)
        return

    @classmethod
    def get_globalVars(cls):
        global_vars = {
            "move_count": cls.move_count,
            "was_checked": cls.was_checked.copy(),
            "is_rochade": cls.is_rochade,
            "is_rochade_gui": cls.is_rochade_gui,
            "rochade_rook": cls.rochade_rook,
            "rochade_move_to": cls.rochade_move_to,
            "rochade_field_idx": cls.rochade_field_idx,
            "queen_counter": cls.queen_counter,
            "is_checkmate": cls.is_checkmate
        }
        return global_vars

    @classmethod
    def load_state(cls, move_counter=False):
        with open(save_file, 'rb') as fin:
            _saved_state = pickle.load(fin)
        return _saved_state

    def move_counter(self):
        Piece.move_count += 1

    def get_field_name(self, field_idx):
        return self.field_names[field_idx[0],field_idx[1]]

    def description(self):

        print("{}, {}".format(self.name, "white" if self.color == 1 else "black"))
        print("Current field: {}".format(self.current_field))
        return

    def get_field_idx(self, field_name):
        return np.where(self.field_names == field_name)

    def valid_field(self, field_to_check, piece_color, board, request_text=True):
        """

        :return: is_valid: 0 if not valid or field occupied by same color,
        1 if empty field,
        2 if field occupied by opposite color
        """
        field_idx = self.get_field_idx(field_to_check)
        field_isvalid = 0
        if board[field_idx][0]:
            field_color = board[field_idx][0].color
            if piece_color == field_color:
                if request_text:
                    print("MoveError: {} contains piece of same color".format(field_to_check))
                pass
            else:
                # print("field {} contains piece of opposite color".format(field_to_check))
                field_isvalid = 2
        else:
            # print("field {} is empty".format(field_to_check))
            field_isvalid = 1
        return field_isvalid

    def get_move_step(self, move_to):
        """returns a tuple of the move vector"""
        # print(tuple([self.get_field_idx(move_to)[i] - self.field_idx[i] for i in range(2)]))
        return tuple([self.get_field_idx(move_to)[i] - self.field_idx[i] for i in range(2)])

    def get_intermediate_steps(self, step, move_to):
        """returns all intermediate steps that need to be valid to successfully execute the move"""
        intermediate_steps = []
        field = [self.field_idx[i][0] for i in range(2)]
        if not "Knight" in self.name:
            steps_number = max([abs(s) for s in step])
            for s in range(steps_number):
                field = [int(field[i] + step[i]/steps_number) for i in range(2)]
                intermediate_steps.append(self.field_names[field[0]][field[1]])
                # print(intermediate_steps)
        else:
            intermediate_steps.append(move_to)
        if "Knight" in self.name:
            intermediate_steps = [intermediate_steps[-1]]
        return intermediate_steps

    def move_validity(self, intermediate_steps, color, request_text=True, is_rochade=False):
        valid_text = ""
        for intermediate_step in intermediate_steps[:-1]:
            # if is_rochade:
            #
            if not self.valid_field(intermediate_step, color, board, request_text) == 1:
                valid_text = "Intermediate step to {} is invalid".format(intermediate_step)
                # print("Intermediate step to {} is invalid".format(intermediate_step))
                return 0, valid_text
        final_step_isvalid = self.valid_field(intermediate_steps[-1], color, board, request_text)
        if not final_step_isvalid:
            return 0, valid_text
        elif final_step_isvalid == 1:
            valid_text = "{} moves to {}".format(self.name, intermediate_steps[-1])
            # print("{} moves to {}".format(self.name, intermediate_steps[-1]))
            gameboard.valid_move = True
            return 1, valid_text
        else:

            enemy_name = board[self.get_field_idx(intermediate_steps[-1])][0].name
            valid_text = "{} kills {} on {}".format(self.name, enemy_name ,intermediate_steps[-1])
            # print("{} kills {} on {}".format(self.name, enemy_name ,intermediate_steps[-1]))
            if "Pawn" in self.name and len(intermediate_steps) > 1:
                if request_text:
                    print("MoveError: {} cannot move to {}".format(self.name, intermediate_steps[-1]))
                return 0, valid_text
            gameboard.valid_move = True
            if request_text:
                gameboard.kill = board[self.get_field_idx(intermediate_steps[-1])][0]
            return 2, valid_text

    def resurrect_piece(self,enemy):
        enemy.is_alive = 1
        enemy.current_field = self.current_field
        enemy.field_idx = self.field_idx
        return

    def piece_move_types(self, move_to):
        self.move_types(move_to)

    def is_check(self, color):
        check = False
        for piece in Piece:
            if "King" in piece.name and piece.color == color:
                move_to = piece.current_field
                king = piece
                break
        request_text = False
        for piece in Piece:
            if piece.color != color and piece.is_alive:
                step = [s[0] for s in piece.get_move_step(move_to)]
                piece.piece_move_types(move_to)
                # print(piece.name, step, piece.possible_moves)
                if step in piece.possible_moves:
                    intermediate_steps = piece.get_intermediate_steps(step, move_to)
                    move_isvalid, valid_text = piece.move_validity(intermediate_steps, piece.color, request_text)
                    if move_isvalid:
                        check = True
                        break
        return check

    def move_piece(self, move_to, move_isvalid, in_rochade=False):

        # empty the current field
        if not in_rochade:
            mate_board = board.copy()
        board[self.field_idx] = None
        # safe current field for possible redo_move
        self.last_field = self.current_field
        # move to requested field
        self.current_field = move_to
        self.field_idx = self.get_field_idx(move_to)
        if move_isvalid == 2:
            self.kill_piece(board[self.field_idx][0])
        board[self.field_idx] = self
        # kill piece on the field you want to move to
        if not in_rochade:
            return mate_board
        else:
            return

    def kill_piece(self, enemy):
        enemy.is_alive = 0
        enemy.current_field = "Out of field"
        enemy.field_idx = None
        return

    def redo_move(self, mate_board, move_isvalid):
        if move_isvalid == 2:
            self.resurrect_piece(mate_board[self.field_idx][0])
        self.current_field = self.last_field
        self.field_idx = self.get_field_idx(self.last_field)

    def check_mate(self, color, mate_checker=True):
        if self.is_check(color):
            print("{} needs to move out of check".format("White" if color == 1 else "Black"))
            for piece in Piece:
                if piece.color == color and piece.is_alive:
                    for i, move_to in np.ndenumerate(self.field_names):
                        piece.move_types(move_to, mate_checker=mate_checker)
                        checkmate = piece.execute_move(move_to, piece.possible_moves, mate_checker)
                        if not checkmate:
                            Piece.was_checked[piece.color -1] = True
                            print("{} can escape check".format("White" if color == 1 else "Black"))
                            return
            Piece.is_checkmate = True
            print("{} is check mate. Congrats!".format("White" if color == 1 else "Black"))
        return

    def check_pawn2queen(self, transformer=False):
        if "Pawn" in self.name and any(row in self.current_field for row in ["1","8"]):
            self.transform, transformer = True, True
            Piece.queen_counter += 1
        return transformer

    def pawn2queen(self):
        color_text = "White" if self.color == 1 else "Black"
        short_name = "Q_" + str(Piece.queen_counter) + color_text[0]
        p = Queen("Queen " + color_text, short_name, self.color, self.current_field,
                  queenw_im if self.color == 1 else queenb_im)
        # board[self.field_idx][0] = None
        board[self.field_idx] = p
        self.kill_piece(self)
        _images[self.short_name] = ""
        _images[p.short_name] = tk.PhotoImage(file=queenw_im if self.color == 1 else queenb_im)
        gameboard.addpiece(p.short_name, _images[p.short_name], p.field_idx)
        mate_board = board.copy()
        return mate_board

    def execute_move(self, move_to, possible_moves, mate_checker=True):
        step = self.get_move_step(move_to)
        step = [s[0] for s in step]
        checkmate = True
        info_text = ""
        request_text = False if not mate_checker else True
        if step in possible_moves:
            # print("Valid Field move of {} to field {}".format(self.name, move_to))
            intermediate_steps = self.get_intermediate_steps(step, move_to)
            move_isvalid, valid_text = self.move_validity(intermediate_steps, self.color, request_text)
            if move_isvalid:
                mate_board = self.move_piece(move_to, move_isvalid)
                if Piece.is_rochade and mate_checker:
                    Piece.rochade_rook.move_piece(Piece.rochade_move_to, 1, in_rochade=True)
                if self.is_check(self.color):
                    self.redo_move(mate_board, move_isvalid)
                    for index, x in np.ndenumerate(board):
                        board[index] = mate_board[index]
                        if board[index]:
                            board[index].field_idx = ([np.array(index[0])], [np.array(index[1])])
                            board[index].current_field = self.field_names[index]
                    info_text = "CheckError: You would be in check position, try again!"
                    Piece.is_rochade_gui, Piece.is_rochade = False, False
                    if mate_checker:
                        gameboard.valid_move = False
                        gameboard.kill = False
                else:
                    if mate_checker:
                        print(valid_text)
                        self.move_idx += 1
                        self.move_counter()
                        if Piece.is_rochade:
                            Piece.rochade_rook.move_idx += 1
                        # Piece.is_rochade = False
                        if self.check_pawn2queen():
                            self.pawn2queen()
                        opp_color = 2 if self.color == 1 else 1
                        self.check_mate(opp_color, mate_checker=False)
                    else:
                        checkmate = False
                        self.redo_move(mate_board, move_isvalid)
                        for index, x in np.ndenumerate(board):
                            board[index] = mate_board[index]
            else:
                pass
        else:
            info_text = "MoveError: {} cannot move to {}".format(self.name, move_to)

        if mate_checker:
            if info_text:
                print(info_text)
            # self.plot_grip()
            return
        else:
            return checkmate

    def plot_grip(self):
        binary_board = np.zeros((8, 8))
        for i in range(len(board)):
            for j in range(len(board[i])):
                element = board[i][j]
                if not element:
                    binary_board[i][j] = 1
                elif element.color == 1:
                    binary_board[i][j] = 2
                else:
                    binary_board[i][j] = 0
        plt.ion()
        plt.pcolormesh(binary_board, cmap="gray", edgecolors='darkgrey')
        plt.plot()
        plt.show(block=False)
        return

    def move(self, move_to):
        print("Move {}:".format(self.move_count))
        move_to = move_to.upper()
        if not self.move_count % 2 == (self.color - 1):
            print("TurnError: It's {}'s turn".format("white" if self.color == 2 else "black"))
            return
        if not self.is_alive:
            print("PieceError: {} was already killed".format(self.name))
            return
        if not np.array(np.where(self.field_names == move_to)).size:
            print("FieldError: The Field doesn't exist! Try another value")
            return
        #self.save_state()
        self.move_types(move_to)
        self.execute_move(move_to, self.possible_moves)
        #self.load_state(self.move_count-1)
        print("")
        return


class Pawn(Piece):
    """
    possible_moves = [v,h]: v = vertical, h = horizontal
    """
    #possible_moves = []
    normal_moves = [[1,0]]
    special_moves = [[2,0]]
    kill_moves = [[1,-1],[1,1]]

    def move(self, move_to):
        super().move(move_to)

    def piece_move_types(self, move_to):
        super().piece_move_types(move_to)

    def move_types(self, move_to, mate_checker=True):
        """changes the possible moves according to the current move_idx"""
        if self.move_idx == 0:
            self.possible_moves = self.normal_moves + self.special_moves + self.kill_moves
        else:
            self.possible_moves = self.normal_moves + self.kill_moves
        if self.color == 2:
            self.possible_moves = [[-i for i in m] for m in self.possible_moves]
        # exclude kill moves if it's not possible
        move_vector = [self.get_field_idx(move_to)[i] - self.field_idx[i] for i in range(2)]
        move_to_idx = self.get_field_idx(move_to)
        kill_move_excluder = False
        if not board[move_to_idx]:
            kill_move_excluder = True
        else:
            if not (move_vector in self.possible_moves[1:] and
                            board[move_to_idx][0].color != self.color):
                kill_move_excluder = True
        if kill_move_excluder:
            self.possible_moves = self.possible_moves[:-2]
        # exclude vertical kills
        vertical_move_excluder = False
        if board[move_to_idx]:
            if move_vector == self.possible_moves[0] and board[move_to_idx][0].color != self.color:
                vertical_move_excluder = True
        if vertical_move_excluder:
            self.possible_moves = self.possible_moves[1:]
        return


class Rook(Piece):
    possible_moves = []

    def move(self, move_to):
        super().move(move_to)

    def piece_move_types(self, move_to):
        super().piece_move_types(move_to)

    def move_types(self, move_to, mate_checker=True):
        move_vector = [self.get_field_idx(move_to)[i] - self.field_idx[i] for i in range(2)]
        if bool(move_vector[0][0]) != bool(move_vector[1][0]):
            self.possible_moves = [[vi[0] for vi in move_vector]]
        return


class Knight(Piece):
    possible_moves = []
    normal_moves = [[2,-1],[2,1],[1,2],[-1,2],[-2,-1],[-2,1],[1,-2],[-1,-2]]

    def move(self, move_to):
        super().move(move_to)

    def piece_move_types(self, move_to):
        super().piece_move_types(move_to)

    def move_types(self, move_to, mate_checker=True):
        self.possible_moves = self.normal_moves
        return


class Bishop(Piece):
    possible_moves = []

    def move(self, move_to):
        super().move(move_to)

    def piece_move_types(self, move_to):
        super().piece_move_types(move_to)

    def move_types(self, move_to, mate_checker=True):
        move_vector = [self.get_field_idx(move_to)[i] - self.field_idx[i] for i in range(2)]
        if abs(move_vector[0][0]) == abs(move_vector[1][0]) and move_vector[0][0] != 0:
            self.possible_moves = [[vi[0] for vi in move_vector]]
        return


class Queen(Piece):
    possible_moves = []

    def move(self, move_to):
        super().move(move_to)

    def piece_move_types(self, move_to):
        super().piece_move_types(move_to)

    def move_types(self, move_to, mate_checker=True):
        move_vector = [self.get_field_idx(move_to)[i] - self.field_idx[i] for i in range(2)]
        if abs(move_vector[0][0]) == abs(move_vector[1][0]) and move_vector[0][0] != 0 or \
                        bool(move_vector[0][0]) != bool(move_vector[1][0]):
            self.possible_moves = [[vi[0] for vi in move_vector]]
        return


class King(Piece):
    possible_moves = []
    normal_moves = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
    rochade = [[0,-2],[0,2]]
    def move(self, move_to):
        super().move(move_to)

    def piece_move_types(self, move_to):
        super().piece_move_types(move_to)

    def move_types(self, move_to, mate_checker=True):
        if self.move_idx == 0:
            self.possible_moves = self.normal_moves
            move_step = [i[0] for i in self.get_move_step(move_to)]
            # print(move_step)
            if move_step in self.rochade and not Piece.was_checked[self.color -1] and mate_checker:
                rooks = [p for p in Piece if "Rook" in p.name and p.color == self.color]
                for rook in rooks:
                    if move_step[-1] > 0 and "2" in rook.name or \
                        move_step[-1] < 0 and "1" in rook.name and rook.move_idx == 0:
                        self.possible_moves = self.normal_moves + [move_step]
                        Piece.is_rochade = True
                        Piece.is_rochade_gui = True
                        Piece.rochade_rook = rook
                        rochade_idx = (self.get_field_idx(move_to)[0][0],
                                       self.get_field_idx(move_to)[1][0] - int(move_step[-1]/2))
                        Piece.rochade_move_to = self.get_field_name(rochade_idx)
        else:
            self.possible_moves = self.normal_moves


def initialize_Board():
    board = np.empty((8, 8), dtype=object)
    column_chars = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for row in range(8):
        for column, column_char in zip(range(8), column_chars):
            field_name = "{}{}".format(column_char, row + 1)
            if list(filter(lambda p: p.current_field == field_name, Piece)):
                p = list(filter(lambda p: p.current_field == field_name, Piece))[0]
                board[row, column] = p
            else:
                pass
    return board

im_path = os.path.join(os.getcwd(), "images")
rookb_im = os.path.join(im_path, "bR.png")
rookw_im = os.path.join(im_path, "wR.png")
pawnb_im = os.path.join(im_path, "bP.png")
pawnw_im = os.path.join(im_path, "wP.png")
knightb_im = os.path.join(im_path, "bN.png")
knightw_im = os.path.join(im_path, "wN.png")
bishopb_im = os.path.join(im_path, "bB.png")
bishopw_im = os.path.join(im_path, "wB.png")
kingb_im = os.path.join(im_path, "bK.png")
kingw_im = os.path.join(im_path, "wK.png")
queenb_im = os.path.join(im_path, "bQ.png")
queenw_im = os.path.join(im_path, "wQ.png")

root = tk.Tk()

P_W1 = Pawn("Pawn White 1", "P_W1", 1, "A2", pawnw_im)
P_W2 = Pawn("Pawn White 2", "P_W2", 1, "B2", pawnw_im)
P_W3 = Pawn("Pawn White 3", "P_W3", 1, "C2", pawnw_im)
P_W4 = Pawn("Pawn White 4", "P_W4", 1, "D2", pawnw_im)
P_W5 = Pawn("Pawn White 5", "P_W5", 1, "E2", pawnw_im)
P_W6 = Pawn("Pawn White 6", "P_W6", 1, "F2", pawnw_im)
P_W7 = Pawn("Pawn White 7", "P_W7", 1, "G2", pawnw_im)
P_W8 = Pawn("Pawn White 8", "P_W8", 1, "H2", pawnw_im)

R_W1 = Rook("Rook White 1", "R_W1", 1, "A1", rookw_im)
R_W2 = Rook("Rook White 2", "R_W2", 1, "H1", rookw_im)

N_W1 = Knight("Knight White 1", "N_W1", 1, "B1", knightw_im)
N_W2 = Knight("Knight White 2", "N_W2", 1, "G1", knightw_im)

B_W1 = Bishop("Bishop White 1", "B_W1", 1, "C1", bishopw_im)
B_W2 = Bishop("Bishop White 2", "B_W2", 1, "F1", bishopw_im)

K_W = King("King White", "K_W", 1, "E1", kingw_im)

Q_W = Queen("Queen White", "Q_W", 1, "D1", queenw_im)


P_B1 = Pawn("Pawn Black 1", "P_B1", 2, "A7", pawnb_im)
P_B2 = Pawn("Pawn Black 2", "P_B2", 2, "B7", pawnb_im)
P_B3 = Pawn("Pawn Black 3", "P_B3", 2, "C7", pawnb_im)
P_B4 = Pawn("Pawn Black 4", "P_B4", 2, "D7", pawnb_im)
P_B5 = Pawn("Pawn Black 5", "P_B5", 2, "E7", pawnb_im)
P_B6 = Pawn("Pawn Black 6", "P_B6", 2, "F7", pawnb_im)
P_B7 = Pawn("Pawn Black 7", "P_B7", 2, "G7", pawnb_im)
P_B8 = Pawn("Pawn Black 8", "P_B8", 2, "H7", pawnb_im)

R_B1 = Rook("Rook Black 1", "R_B1", 2, "A8", rookb_im)
R_B2 = Rook("Rook Black 2", "R_B2", 2, "H8", rookb_im)

N_B1 = Knight("Knight Black 1", "N_B1", 2, "B8", knightb_im)
N_B2 = Knight("Knight Black 2", "N_B2", 2, "G8", knightb_im)

B_B1 = Bishop("Bishop Black 1", "B_B1", 2, "C8", bishopb_im)
B_B2 = Bishop("Bishop Black 2", "B_B2", 2, "F8", bishopb_im)

K_B = King("King Black", "K_B", 2, "E8", kingb_im)

Q_B = Queen("Queen Black", "Q_B", 2, "D8", queenb_im)

save_file = os.path.join(os.getcwd(), "saved_state.pkl")

board = initialize_Board()
Piece.save_state(initializer=True)
gameboard = GameBoard(root)
gameboard.pack(side="top", fill="both", expand="False", padx=0, pady=0)
_images = dict()
for p in Piece:
    _images[p.short_name] = tk.PhotoImage(file=p.img_file)
    gameboard.addpiece(p.short_name, _images[p.short_name], p.field_idx)
root.mainloop()

os.remove(save_file)

