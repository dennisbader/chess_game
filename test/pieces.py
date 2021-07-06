import os
import numpy as np

import tkinter as tk
import _pickle as pickle


class IterRegistry(type):
    def __iter__(cls):
        return iter(cls._registry)


class Piece(metaclass=IterRegistry):
    _registry = []
    column_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
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
        self.field_names = np.array([['{}{}'.format(i,j) for i in self.column_chars] for j in range(1,8+1)])
        self.field_idx = np.where(self.field_names == self.current_field)
        self.move_idx = 0
        self.is_alive = 1
        self.possible_moves = []
        self.checks = False
        self.img_file = img_file
        self.transform = False

    @classmethod
    def save_state(cls, initializer=False):
        move_count = -1 if initializer else cls.move_count - 1
        if os.path.exists(save_file):
            saved_state = cls.load_state()
            os.remove(save_file)
        else:
            saved_state = dict()
        saved_state[move_count] = {
            'board': board.copy(),
            'pieces': [p for p in Piece],
            'globalVars': cls.get_globalVars()
        }

        with open(save_file, 'ab') as fout:
            pickle.dump(saved_state, fout, -1)
        return

    @classmethod
    def get_globalVars(cls):
        global_vars = {
            'move_count': cls.move_count,
            'was_checked': cls.was_checked.copy(),
            'is_rochade': cls.is_rochade,
            'is_rochade_gui': cls.is_rochade_gui,
            'rochade_rook': cls.rochade_rook,
            'rochade_move_to': cls.rochade_move_to,
            'rochade_field_idx': cls.rochade_field_idx,
            'queen_counter': cls.queen_counter,
            'is_checkmate': cls.is_checkmate
        }
        return global_vars

    @classmethod
    def load_state(cls, move_counter=False):
        with open(save_file, 'rb') as fin:
            _saved_state = pickle.load(fin)
        return _saved_state

    @staticmethod
    def move_counter():
        Piece.move_count += 1

    def get_field_name(self, field_idx):
        return self.field_names[field_idx[0],field_idx[1]]

    def description(self):
        print('{}, {}'.format(self.name, 'white' if self.color == 1 else 'black'))
        print('Current field: {}'.format(self.current_field))
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
                    print('MoveError: {} contains piece of same color'.format(field_to_check))
                pass
            else:
                field_isvalid = 2
        else:
            field_isvalid = 1
        return field_isvalid

    def get_move_step(self, move_to):
        """returns a tuple of the move vector"""
        return tuple([self.get_field_idx(move_to)[i] - self.field_idx[i] for i in range(2)])

    def get_intermediate_steps(self, step, move_to):
        """returns all intermediate steps that need to be valid to successfully execute the move"""
        intermediate_steps = []
        field = [self.field_idx[i][0] for i in range(2)]
        if not 'Knight' in self.name:
            steps_number = max([abs(s) for s in step])
            for s in range(steps_number):
                field = [int(field[i] + step[i]/steps_number) for i in range(2)]
                intermediate_steps.append(self.field_names[field[0]][field[1]])
        else:
            intermediate_steps.append(move_to)
        if 'Knight' in self.name:
            intermediate_steps = [intermediate_steps[-1]]
        return intermediate_steps

    def move_validity(self, intermediate_steps, color, request_text=True):
        valid_text = ''
        for intermediate_step in intermediate_steps[:-1]:
            if not self.valid_field(intermediate_step, color, board, request_text) == 1:
                valid_text = 'Intermediate step to {} is invalid'.format(intermediate_step)
                return 0, valid_text
        final_step_isvalid = self.valid_field(intermediate_steps[-1], color, board, request_text)
        if not final_step_isvalid:
            return 0, valid_text
        elif final_step_isvalid == 1:
            valid_text = '{} moves to {}'.format(self.name, intermediate_steps[-1])
            gameboard.valid_move = True
            return 1, valid_text
        else:

            enemy_name = board[self.get_field_idx(intermediate_steps[-1])][0].name
            valid_text = '{} kills {} on {}'.format(self.name, enemy_name ,intermediate_steps[-1])
            if 'Pawn' in self.name and len(intermediate_steps) > 1:
                if request_text:
                    print('MoveError: {} cannot move to {}'.format(self.name, intermediate_steps[-1]))
                return 0, valid_text
            gameboard.valid_move = True
            if request_text:
                gameboard.kill = board[self.get_field_idx(intermediate_steps[-1])][0]
            return 2, valid_text

    def resurrect_piece(self, enemy):
        enemy.is_alive = 1
        enemy.current_field = self.current_field
        enemy.field_idx = self.field_idx
        return

    def piece_move_types(self, move_to):
        self.move_types(move_to)

    def is_check(self, color):
        check = False
        for piece in Piece:
            if 'King' in piece.name and piece.color == color:
                move_to = piece.current_field
                king = piece
                break
        request_text = False
        for piece in Piece:
            if piece.color != color and piece.is_alive:
                step = [s[0] for s in piece.get_move_step(move_to)]
                piece.piece_move_types(move_to)
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

    @staticmethod
    def kill_piece(enemy):
        enemy.is_alive = 0
        enemy.current_field = 'Out of field'
        enemy.field_idx = None
        return

    def redo_move(self, mate_board, move_isvalid):
        if move_isvalid == 2:
            self.resurrect_piece(mate_board[self.field_idx][0])
        self.current_field = self.last_field
        self.field_idx = self.get_field_idx(self.last_field)

    def check_mate(self, color, mate_checker=True):
        if self.is_check(color):
            print('{} needs to move out of check'.format('White' if color == 1 else 'Black'))
            for piece in Piece:
                if piece.color == color and piece.is_alive:
                    for i, move_to in np.ndenumerate(self.field_names):
                        piece.move_types(move_to, mate_checker=mate_checker)
                        checkmate = piece.execute_move(move_to, piece.possible_moves, mate_checker)
                        if not checkmate:
                            Piece.was_checked[piece.color -1] = True
                            print('{} can escape check'.format('White' if color == 1 else 'Black'))
                            return
            Piece.is_checkmate = True
            print('{} is check mate. Congrats!'.format('White' if color == 1 else 'Black'))
        return

    def check_pawn2queen(self, transformer=False):
        if 'Pawn' in self.name and any(row in self.current_field for row in ['1','8']):
            self.transform, transformer = True, True
            Piece.queen_counter += 1
        return transformer

    def pawn2queen(self):
        color_text = 'White' if self.color == 1 else 'Black'
        short_name = 'Q_' + str(Piece.queen_counter) + color_text[0]
        p = Queen('Queen ' + color_text, short_name, self.color, self.current_field,
                  queenw_im if self.color == 1 else queenb_im)
        board[self.field_idx] = p
        self.kill_piece(self)
        _images[self.short_name] = ''
        _images[p.short_name] = tk.PhotoImage(file=queenw_im if self.color == 1 else queenb_im)
        gameboard.addpiece(p.short_name, _images[p.short_name], p.field_idx)
        mate_board = board.copy()
        return mate_board

    def execute_move(self, move_to, possible_moves, mate_checker=True):
        step = self.get_move_step(move_to)
        step = [s[0] for s in step]
        checkmate = True
        info_text = ''
        request_text = False if not mate_checker else True
        if step in possible_moves:
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
                    info_text = 'CheckError: You would be in check position, try again!'
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
            info_text = 'MoveError: {} cannot move to {}'.format(self.name, move_to)

        if mate_checker:
            if info_text:
                print(info_text)
            return
        else:
            return checkmate

    def move(self, move_to):
        print('Move {}:'.format(self.move_count))
        move_to = move_to.upper()
        if not self.move_count % 2 == (self.color - 1):
            print("TurnError: It is {}'s turn".format('white' if self.color == 2 else 'black'))
            return
        if not self.is_alive:
            print('PieceError: {} was already killed'.format(self.name))
            return
        if not np.array(np.where(self.field_names == move_to)).size:
            print('FieldError: The Field does not exist! Try another value')
            return
        self.move_types(move_to)
        self.execute_move(move_to, self.possible_moves)
        print('')
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
            if move_step in self.rochade and not Piece.was_checked[self.color -1] and mate_checker:
                rooks = [p for p in Piece if 'Rook' in p.name and p.color == self.color]
                for rook in rooks:
                    if move_step[-1] > 0 and '2' in rook.name \
                            or move_step[-1] < 0 and '1' in rook.name and rook.move_idx == 0:
                        self.possible_moves = self.normal_moves + [move_step]
                        Piece.is_rochade = True
                        Piece.is_rochade_gui = True
                        Piece.rochade_rook = rook
                        rochade_idx = (self.get_field_idx(move_to)[0][0],
                                       self.get_field_idx(move_to)[1][0] - int(move_step[-1]/2))
                        Piece.rochade_move_to = self.get_field_name(rochade_idx)
        else:
            self.possible_moves = self.normal_moves
