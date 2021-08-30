import os
import abc
import numpy as np

import _pickle as pickle

from text_styles import TextStyle


class IterRegistry(type):
    def __iter__(cls):
        return iter(cls.pieces)


class GameOps(metaclass=IterRegistry):
    """Contains the entire game logic and piece objects
    """

    pieces = []
    column_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    row_chars = [str(i) for i in range(1, len(column_chars) + 1)]
    move_count = 0
    verbose = True
    board_gui = None
    board_objects = None
    # saves if white/black were already in check (for rochade): idx 0 -> white, idx 1 -> black]
    was_checked = [False, False]
    is_rochade, is_rochade_gui = False, False
    rochade_rook = None
    rochade_rook_move_to = None
    rochade_field_idx = None
    queen_counter = 0
    is_checkmate = False
    save_file = os.path.join(os.path.dirname(__file__), 'saved_state.pkl')

    def __init__(self, name, short_name, color, current_field, img_file):
        """Creates a chess piece object
        Arguments:
            name: piece name long
            short_name: piece name short
            color: 1 for white, 2 for black
            current_field: piece's start field
            img_file: path to the piece's image
        """
        self.pieces.append(self)
        self.name = name
        self.short_name = short_name
        self.color = color
        self.img_file = img_file
        self.is_alive = True
        self.checks = False
        self.promote = False  # promote pawn into another piece

        self.field_names = self.get_field_names(row_chars=self.row_chars, column_chars=self.column_chars)
        self.current_field = current_field
        self.last_field = current_field
        self.field_idx = self.get_field_idx(self.current_field)
        self.possible_moves = []
        self.receive_en_passant = False
        self.perform_en_passant = False
        self.en_passant_idx = (None, None)

        self.move_idx = 0

    @staticmethod
    def get_field_names(row_chars, column_chars):
        return np.array([['{}{}'.format(col, row) for col in column_chars] for row in row_chars])

    @classmethod
    def initialize_game(cls, board_objects, board_gui):
        cls.board_objects = board_objects
        cls.board_gui = board_gui
        cls.save_state(initializer=True)

    @classmethod
    def save_state(cls, initializer=False):
        move_count = -1 if initializer else cls.move_count - 1
        if os.path.exists(cls.save_file):
            saved_state = cls.load_state()
            os.remove(cls.save_file)
        else:
            saved_state = dict()
        saved_state[move_count] = {
            'board': cls.board_objects.copy(),
            'pieces': [piece for piece in GameOps],
            'globalVars': cls.get_global_vars()
        }

        with open(cls.save_file, 'ab') as f_out:
            pickle.dump(saved_state, f_out, -1)
        return

    @classmethod
    def load_state(cls, move_counter=False):
        with open(cls.save_file, 'rb') as f_in:
            _saved_state = pickle.load(f_in)
        return _saved_state

    @classmethod
    def get_global_vars(cls):
        global_vars = {
            'move_count': cls.move_count,
            'was_checked': cls.was_checked.copy(),
            'is_rochade': cls.is_rochade,
            'is_rochade_gui': cls.is_rochade_gui,
            'rochade_rook': cls.rochade_rook,
            'rochade_rook_move_to': cls.rochade_rook_move_to,
            'rochade_field_idx': cls.rochade_field_idx,
            'queen_counter': cls.queen_counter,
            'is_checkmate': cls.is_checkmate
        }
        return global_vars

    @staticmethod
    def move_counter():
        GameOps.move_count += 1

    def get_field_name(self, field_idx):
        return self.field_names[field_idx]

    def get_field_idx(self, target_field):
        field_idx = np.where(self.field_names == target_field)
        return tuple([field_idx[0][0], field_idx[1][0]])

    @classmethod
    def output_text(cls, text, prefix=None, style='black'):
        if not cls.verbose:
            return

        # text
        text = '{} {}\n'.format(prefix if prefix is not None else str(cls.move_count) + '.', text)
        text = text.lower()

        # text style
        launcher = 'gui' if cls.board_gui is not None else 'console'
        style = TextStyle.styles[launcher][style]
        style_end = TextStyle.styles[launcher]['end']

        if cls.board_gui is not None:  # print in GUI
            cls.board_gui.insert_text(cls.board_gui.widgets['text']['info'], text, style=style)
        else:  # print in console
            text = style + text + style_end
            print(text)

    def description(self):
        self.output_text('{}, {}'.format(self.name, 'white' if self.color == 1 else 'black'))
        self.output_text('Current field: {}'.format(self.current_field))
        return

    def valid_field(self, field_to_check, piece_color, board_objects, is_test=False):
        """
        :return: field validity with values
            0:  if not valid or field occupied by same color,
            1:  if empty field,
            2:  if field occupied by opposite color
        """
        field_idx = self.get_field_idx(field_to_check)
        field_isvalid = 0
        if isinstance(board_objects[field_idx], GameOps):  # field contains a piece
            field_color = board_objects[field_idx].color
            if piece_color == field_color:  # piece cannot move to field occupied by same color
                if not is_test:
                    self.output_text(
                        text='MoveError: {} contains piece of same color'.format(field_to_check),
                        style='warning')
            else:
                field_isvalid = 2  # piece kills opposite color
        else:
            if self.perform_en_passant:  # en passant has same functionality as killing
                field_isvalid = 2
            else:  # field is empty and valid
                field_isvalid = 1
        return field_isvalid

    def get_move_step(self, move_to):
        """returns a tuple of the move vector"""
        move_to_idx = self.get_field_idx(move_to)
        return tuple([move_to_idx[i] - self.field_idx[i] for i in range(2)])

    def get_intermediate_field_names(self, move_to):
        """returns all intermediate steps that need to be valid to successfully execute the move"""
        field_idx = self.field_idx
        move_vector = list(self.get_move_step(move_to))
        step_count = max([abs(s) for s in move_vector])

        intermediate_steps = []
        for s in range(step_count):  # identifies all intermediate field names
            field_idx = tuple([int(field_idx[i] + move_vector[i]/step_count) for i in range(2)])
            intermediate_steps.append(self.get_field_name(field_idx))

        if 'Knight' in self.name:  # Knight jumps over
            intermediate_steps = [move_to]
        return intermediate_steps

    def move_validity(self, move_to, color, is_test=False):
        """
            0:  at least one of the intermediate steps (or final step) is invalid
            1:  move is valid
            2:  move is valid and piece kills another piece
        """
        valid_text = ''
        valid_code = 0
        intermediate_steps = self.get_intermediate_field_names(move_to)

        for intermediate_step in intermediate_steps[:-1]:
            if not self.valid_field(intermediate_step, color, self.board_objects, is_test) == 1:
                valid_text = 'Intermediate step to {} is invalid'.format(intermediate_step)
                return valid_code, valid_text

        final_step_isvalid = self.valid_field(intermediate_steps[-1], color, self.board_objects, is_test)
        if final_step_isvalid == 0:  # invalid destination field or occupied by same color
            valid_code = 0
        elif final_step_isvalid == 1:  # empty field
            valid_text = '{} moves to {}'.format(self.name, intermediate_steps[-1])
            valid_code = 1
            self.board_gui.valid_move = True
        elif final_step_isvalid == 2:  # field occupied by opposite color
            if self.perform_en_passant:  # en passant kill
                enemy_idx = self.en_passant_idx
            else:  # normal kill
                enemy_idx = self.get_field_idx(intermediate_steps[-1])

            enemy_name = self.board_objects[enemy_idx].name
            valid_text = '{} kills {} on {}'.format(self.name, enemy_name, intermediate_steps[-1])
            valid_code = 2
            self.board_gui.valid_move = True
            if not is_test:
                self.board_gui.kill = self.board_objects[enemy_idx]
        else:  # unknown field validity code
            raise ValueError('Unknown field validity: {}'.format(final_step_isvalid))

        return valid_code, valid_text

    def resurrect_piece(self, enemy):
        enemy.is_alive = True
        enemy.current_field = self.current_field
        enemy.field_idx = self.field_idx
        return

    @staticmethod
    def is_check(color):
        check = False
        for piece in GameOps:
            if 'King' in piece.name and piece.color == color:
                move_to = piece.current_field
                break
        is_test = True
        for piece in GameOps:
            if piece.color != color and piece.is_alive:
                step = list(piece.get_move_step(move_to))
                piece.piece_move_types(move_to, is_test=is_test)
                if step in piece.possible_moves:
                    move_isvalid, valid_text = piece.move_validity(move_to, piece.color, is_test)
                    if move_isvalid:
                        check = True
                        break
        return check

    def move_piece(self, move_to, move_isvalid, in_rochade=False):
        # empty the current field
        if not in_rochade:
            mate_board = self.board_objects.copy()

        self.board_objects[self.field_idx] = None
        # safe current field for possible redo_move
        self.last_field = self.current_field
        # move to requested field
        self.current_field = move_to
        self.field_idx = self.get_field_idx(move_to)

        if move_isvalid == 2:  # piece kills
            if self.perform_en_passant:
                self.kill_piece(self.board_objects[self.en_passant_idx])
            else:
                self.kill_piece(self.board_objects[self.field_idx])

        self.board_objects[self.field_idx] = self
        # kill piece on the field you want to move to
        if not in_rochade:
            return mate_board
        else:
            return

    def kill_piece(self, enemy, replace=False):
        if not replace:
            self.board_objects[enemy.field_idx] = None
        enemy.is_alive = False
        enemy.current_field = 'Out of field'
        enemy.field_idx = None
        return

    def redo_move(self, mate_board, move_isvalid):
        if move_isvalid == 2:
            self.resurrect_piece(mate_board[self.field_idx])
        self.current_field = self.last_field
        self.field_idx = self.get_field_idx(self.last_field)

    def check_mate(self, color, mate_checker=True):
        if self.is_check(color):
            self.output_text(
                text='{} needs to move out of check'.format('White' if color == 1 else 'Black'),
                prefix='--', style='warning')
            for piece in GameOps:
                if piece.color == color and piece.is_alive:
                    for i, move_to in np.ndenumerate(self.field_names):
                        piece.move_types(move_to, mate_checker=mate_checker, is_test=True)
                        checkmate = piece.execute_move(move_to, piece.possible_moves, mate_checker)
                        if not checkmate:
                            GameOps.was_checked[piece.color - 1] = True
                            self.output_text(
                                text='{} can escape check'.format('White' if color == 1 else 'Black'),
                                prefix='--', style='normal')
                            return
            GameOps.is_checkmate = True
            self.output_text(
                text='{} is check mate. Congrats!'.format('White' if color == 1 else 'Black'),
                prefix='**', style='win')
        return

    def check_pawn2queen(self, promoter=False):
        if 'Pawn' in self.name and any(row in self.current_field for row in ['1','8']):
            self.promote, promoter = True, True
            GameOps.queen_counter += 1
        return promoter

    def check_en_passant(self):
        pass

    def pawn_en_passant(self):
        pass

    def reset_en_passant(self):
        """resets the possibility to perform en passant on pawns after move was executed"""
        for piece in GameOps.pieces:
            if not piece == self:
                piece.receive_en_passant = False
        return

    def pawn2queen(self):
        color_text = 'White' if self.color == 1 else 'Black'
        short_name = 'Q_' + str(GameOps.queen_counter) + color_text[0]
        queen_white_fin = self.board_gui.image_paths['Q_W']
        queen_black_fin = self.board_gui.image_paths['Q_B']
        p = Queen('Queen ' + color_text, short_name, self.color, self.current_field,
                  queen_white_fin if self.color == 1 else queen_black_fin)
        self.board_objects[self.field_idx] = p
        self.kill_piece(self, replace=True)
        self.board_gui.images['pieces'][self.short_name] = ''
        self.board_gui.images['pieces'][p.short_name] \
            = self.board_gui.read_image(fin=queen_white_fin if self.color == 1 else queen_black_fin)
        # self.board_gui.images['pieces'][p.short_name] \
        #     = tk.PhotoImage(file=queen_white_fin if self.color == 1 else queen_black_fin)
        self.board_gui.add_piece(p.short_name, self.board_gui.images['pieces'][p.short_name], p.field_idx)
        mate_board = self.board_objects.copy()
        return mate_board

    def execute_move(self, move_to, possible_moves, mate_checker=True):
        step = list(self.get_move_step(move_to))
        checkmate = True
        info_text = ''
        is_test = True if not mate_checker else False
        if step in possible_moves:  # move is in piece's move set
            move_isvalid, valid_text = self.move_validity(move_to, self.color, is_test)
            if move_isvalid:
                mate_board = self.move_piece(move_to, move_isvalid)
                if GameOps.is_rochade and mate_checker:
                    GameOps.rochade_rook.move_piece(GameOps.rochade_rook_move_to, 1, in_rochade=True)
                if self.is_check(self.color):
                    self.redo_move(mate_board, move_isvalid)
                    for index, x in np.ndenumerate(self.board_objects):
                        self.board_objects[index] = mate_board[index]
                        if self.board_objects[index]:
                            self.board_objects[index].field_idx = index
                            self.board_objects[index].current_field = self.field_names[index]
                    info_text = 'CheckError: You would be in check position, try again!'
                    GameOps.is_rochade_gui, GameOps.is_rochade = False, False
                    if mate_checker:
                        self.board_gui.valid_move = False
                        self.board_gui.kill = None
                else:
                    if mate_checker:
                        self.output_text(valid_text)
                        self.move_idx += 1
                        self.move_counter()
                        if GameOps.is_rochade:
                            GameOps.rochade_rook.move_idx += 1
                        if self.check_pawn2queen():
                            self.pawn2queen()
                        opp_color = 2 if self.color == 1 else 1
                        self.check_mate(opp_color, mate_checker=False)
                    else:
                        checkmate = False
                        self.redo_move(mate_board, move_isvalid)
                        for index, x in np.ndenumerate(self.board_objects):
                            self.board_objects[index] = mate_board[index]
            else:
                pass
        else:
            info_text = 'MoveError: {} cannot move to {}'.format(self.name, move_to)

        if mate_checker:
            if info_text:
                self.output_text(info_text, prefix='--', style='warning')
            return
        else:
            return checkmate

    def move(self, move_to):
        move_to = move_to.upper()

        warning = ''
        if not self.move_count % 2 == (self.color - 1):
            warning = "TurnError: It is {}'s turn".format('white' if self.color == 2 else 'black')
        if not self.is_alive:
            warning = 'PieceError: {} was already killed'.format(self.name)
        if not np.array(np.where(self.field_names == move_to)).size:
            warning = 'FieldError: The Field does not exist! Try another value',

        if warning:
            self.output_text(text=warning, style='warning')
            return

        self.move_types(move_to, is_test=False)
        self.execute_move(move_to, self.possible_moves)
        self.reset_en_passant()
        return

    def piece_move_types(self, move_to, is_test=False):
        self.move_types(move_to, is_test=is_test)

    @abc.abstractmethod
    def move_types(self, move_to, is_test=False):
        """defines the move set of a specific chess piece"""
        pass

    @staticmethod
    def is_diagonal(move_vector):
        """diagonal with same steps in vertical as horizontal direction"""
        move_v, move_h = move_vector
        return abs(move_v) == abs(move_h) and move_v != 0

    @staticmethod
    def is_straight(move_vector):
        """either vertical or horizontal direction"""
        move_v, move_h = move_vector
        return bool(move_v) != bool(move_h)

    @staticmethod
    def is_vertical(move_vector):
        """vertical direction"""
        move_v, move_h = move_vector
        return move_v != 0 and move_h == 0

    @staticmethod
    def is_horizontal(move_vector):
        """horizontal direction"""
        move_v, move_h = move_vector
        return move_v == 0 and move_h != 0


class Pawn(GameOps):
    """Pawn has different move sets depending on the current game situation:
        1) if Pawn has not moved yet: Pawn can move one or two steps vertically
        2) if Pawn has already moved: Pawn can move one step vertically
        3) Pawn can onl kill in a one-step positive (relative to the black/white) diagonal direction
    """
    normal_moves = [[1, 0]]
    special_moves = [[2, 0]]
    kill_moves = [[1, -1], [1, 1]]

    def move_types(self, move_to, mate_checker=True, is_test=False):
        if self.move_idx == 0:  # case 1)
            self.possible_moves = self.normal_moves + self.special_moves + self.kill_moves
        else:  # case 2)
            self.possible_moves = self.normal_moves + self.kill_moves
        if self.color == 2:  # black
            self.possible_moves = [[-i for i in m] for m in self.possible_moves]

        # exclude invalid moves
        kill_moves = self.possible_moves[-2:]
        move_vector = list(self.get_move_step(move_to))
        move_to_idx = self.get_field_idx(move_to)
        move_valid = True

        self.perform_en_passant = False
        if abs(move_vector[0]) == 2 and not is_test:
            self.receive_en_passant = True

        if move_vector in kill_moves:
            # Pawn can only move diagonally if it kills another Piece
            if self.board_objects[move_to_idx] is None:
                move_valid = False

                # check whether en passant can be performed
                en_passant_idx = (self.field_idx[0], move_to_idx[1])
                en_passant_piece = self.board_objects[en_passant_idx]
                if isinstance(en_passant_piece, Pawn):
                    if en_passant_piece.receive_en_passant and en_passant_piece.color != self.color:
                        move_valid = True
                        self.perform_en_passant = True
                        self.en_passant_idx = en_passant_idx
        else:
            # Pawn can only move if field is not occupied
            if self.board_objects[move_to_idx] is not None:
                move_valid = False

        if not move_valid:
            self.possible_moves = []
        return


class Rook(GameOps):
    """Rook can move either horizontally or vertically (straight)"""

    def move_types(self, move_to, mate_checker=True, is_test=False):
        move_vector = list(self.get_move_step(move_to))
        if self.is_straight(move_vector):  # vertical-only or horizontal-only
            self.possible_moves = [move_vector]
        return


class Knight(GameOps):
    """Knight has eight possible moves (with different signs) coming from two major move types:
        1) Two horizontal steps, one vertical step
        1) Two vertical steps, one horizontal step
    """

    normal_moves = [[2, -1], [2, 1], [1, 2], [-1, 2], [-2, -1], [-2, 1], [1, -2], [-1, -2]]

    def move_types(self, move_to, mate_checker=True, is_test=False):
        self.possible_moves = self.normal_moves
        return


class Bishop(GameOps):
    """Bishop can move diagonally"""

    def move_types(self, move_to, mate_checker=True, is_test=False):
        move_vector = list(self.get_move_step(move_to))
        if self.is_diagonal(move_vector):
            self.possible_moves = [move_vector]
        return


class Queen(GameOps):
    """Queen can move in any direction (diagonal, vertical, horizontal)"""

    def move_types(self, move_to, mate_checker=True, is_test=False):
        move_vector = list(self.get_move_step(move_to))
        if self.is_diagonal(move_vector) or self.is_straight(move_vector):
            self.possible_moves = [move_vector]
        return


class King(GameOps):
    """King can has two move types:
        1) move one step in any direction
        2) King can do a rochade once if King has not been in check, has not moved yet and rochade is a valid move
    """

    normal_moves = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    rochade = [[0, -2], [0, 2]]

    def move_types(self, move_to, mate_checker=True, is_test=False):
        self.possible_moves = self.normal_moves
        if self.move_idx == 0:
            move_vector = list(self.get_move_step(move_to))
            is_rochade, rook_rochade = self.check_rochade(move_vector, mate_checker)
            if is_rochade:
                self.possible_moves = self.normal_moves + [move_vector]
                self.set_rochade(move_vector, rook_rochade)
        return

    def check_rochade(self, move_vector, mate_checker, is_test=False):
        """checks whether the move is a valid rochade"""
        move_v, move_h = move_vector
        is_rochade, rook_rochade = False, None
        if move_vector in self.rochade and not GameOps.was_checked[self.color - 1] and mate_checker:
            rooks = [p for p in GameOps if 'Rook' in p.name and p.color == self.color and p.is_alive]
            for rook in rooks:
                if (move_h > 0 and '2' in rook.name) or (move_h < 0 and '1' in rook.name) and rook.move_idx == 0:
                    is_rochade = True
                    rook_rochade = rook
                    break
        return is_rochade, rook_rochade

    def set_rochade(self, move_vector, rook_rochade, is_test=False):
        """sets all required settings for the rochade"""
        GameOps.is_rochade = True
        GameOps.is_rochade_gui = True
        GameOps.rochade_rook = rook_rochade
        rochade_idx = (self.field_idx[0], self.field_idx[1] + int(move_vector[1] / 2))
        GameOps.rochade_rook_move_to = self.get_field_name(rochade_idx)
        return
