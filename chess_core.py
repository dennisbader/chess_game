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
    field_names = None
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
    codes = {
        'invalid': 0,
        'empty': 1,
        'opponent': 2,
    }

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

        if GameOps.field_names is None:  # field names only have to be created once
            GameOps.field_names = GameOps.get_field_names(row_chars=self.row_chars, column_chars=self.column_chars)
        self.current_field = current_field
        self.last_field = current_field
        self.field_idx = self.label2index(self.current_field)
        self.possible_moves = []
        self.receive_en_passant = False
        self.perform_en_passant = False
        self.en_passant_idx = (None, None)

        self.move_idx = 0

    @staticmethod
    def get_field_names(row_chars, column_chars):
        return np.array([['{}{}'.format(col, row) for col in column_chars] for row in row_chars])

    @classmethod
    def index2label(cls, field_idx):
        """converts a field index to field label (name)"""

        return cls.field_names[field_idx]

    @classmethod
    def label2index(cls, label):
        """converts a field label (name) to field index"""

        field_idx = np.where(cls.field_names == label)
        return tuple([field_idx[0][0], field_idx[1][0]])

    @classmethod
    def initialize_game(cls, board_objects, board_gui):
        """initializes the game by registering the board objects (pieces), the GUI board and initializing the save file
        """
        cls.board_objects = board_objects
        cls.board_gui = board_gui
        cls.save_state(initializer=True)

    @classmethod
    def save_state(cls, initializer=False):
        """saves the current game state"""

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
        """loads the game's save file"""

        with open(cls.save_file, 'rb') as f_in:
            _saved_state = pickle.load(f_in)
        return _saved_state

    @classmethod
    def get_global_vars(cls):
        """gets the required game status values to save and retrieve specific moves"""

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
        """increments the move count after valid move"""

        GameOps.move_count += 1

    @classmethod
    def output_text(cls, text, prefix=None, style='black'):
        """outputs gane info text (log) into one of the following:
            -   GUI if game is played with the GUI
            -   console if otherwise
        """

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

    def valid_field(self, field_to_check, piece_color, board_objects, is_test=False):
        """
        :return: field validity with values
            0:  if not valid or field occupied by same color,
            1:  if empty field,
            2:  if field occupied by opposite color
        """

        field_idx = self.label2index(field_to_check)
        validity = self.codes['invalid']
        if isinstance(board_objects[field_idx], GameOps):  # field contains a piece
            field_color = board_objects[field_idx].color
            if piece_color == field_color:  # piece cannot move to field occupied by same color
                if not is_test:
                    self.output_text(
                        text='MoveError: {} contains piece of same color'.format(field_to_check),
                        style='warning')
            else:
                validity = self.codes['opponent']  # piece kills opposite color
        else:
            if self.perform_en_passant:  # en passant has same functionality as killing
                validity = self.codes['opponent']
            else:  # field is empty and valid
                validity = self.codes['empty']
        return validity

    def get_move_step(self, move_to):
        """returns a tuple of the move vector"""

        move_to_idx = self.label2index(move_to)
        return tuple([move_to_idx[i] - self.field_idx[i] for i in range(2)])

    def get_intermediate_field_names(self, move_to):
        """returns all intermediate steps that need to be valid to successfully execute the move"""

        field_idx = self.field_idx
        move_vector = list(self.get_move_step(move_to))
        step_count = max([abs(s) for s in move_vector])

        intermediate_steps = []
        for s in range(step_count):  # identifies all intermediate field names
            field_idx = tuple([int(field_idx[i] + move_vector[i]/step_count) for i in range(2)])
            intermediate_steps.append(self.index2label(field_idx))

        if 'Knight' in self.name:  # Knight jumps over
            intermediate_steps = [move_to]
        return intermediate_steps

    def check_move_validity(self, move_to, color, is_test=False):
        """
            0:  -   move is not in piece's move set or
                -   at least one of the intermediate steps or
                -   final step is invalid
            1:  move is valid to an empty field
            2:  move is valid and piece kills another piece
        """

        valid_text = ''
        end_code = self.codes['invalid']

        self.move_types(move_to, is_test=is_test)
        step = list(self.get_move_step(move_to))
        if step not in self.possible_moves:
            valid_text = 'MoveError: {} cannot move to {}'.format(self.short_name, move_to)
            return end_code, valid_text

        steps = self.get_intermediate_field_names(move_to)

        for step in steps[:-1]:  # intermediate steps before end field
            if not self.valid_field(step, color, self.board_objects, is_test) == self.codes['empty']:
                valid_text = 'Intermediate step to {} is invalid'.format(step)
                return end_code, valid_text

        end_code = self.valid_field(steps[-1], color, self.board_objects, is_test)
        if end_code == self.codes['invalid']:  # invalid destination field or occupied by same color
            pass
        elif end_code == self.codes['empty']:  # empty field
            valid_text = '{} moves to {}'.format(self.short_name, steps[-1])
            self.board_gui.valid_move = True
        elif end_code == self.codes['opponent']:  # field occupied by opposite color
            if self.perform_en_passant:  # en passant kill
                enemy_idx = self.en_passant_idx
            else:  # normal kill
                enemy_idx = self.label2index(steps[-1])

            enemy_short_name = self.board_objects[enemy_idx].short_name
            valid_text = '{} kills {} on {}'.format(self.short_name, enemy_short_name, steps[-1])
            self.board_gui.valid_move = True
            if not is_test:
                self.board_gui.kill = self.board_objects[enemy_idx]
        else:  # unknown field validity code
            raise ValueError('Unknown field validity: {}'.format(end_code))

        return end_code, valid_text

    def check_pawn2queen(self, promoter=False):
        """checks whether pawn can be promoted"""

        if 'Pawn' in self.name and any(row in self.current_field for row in ['1', '8']):
            self.promote, promoter = True, True
            GameOps.queen_counter += 1
        return promoter

    def move_piece(self, move_to, move_validity, in_rochade=False):
        """performs the move with one of the following:
            -   rochade
            -   move to empty field
            -   kill normal
            -   kill en passant
        """
        if not in_rochade:
            mate_board = self.board_objects.copy()

        self.board_objects[self.field_idx] = None  # empty the current field
        self.last_field = self.current_field  # safe current field for possible redo_move
        self.current_field = move_to  # move to requested field
        self.field_idx = self.label2index(move_to)

        if move_validity == self.codes['opponent']:  # piece kills
            if self.perform_en_passant:  # with en passant
                self.kill_piece(self.board_objects[self.en_passant_idx])
            else:  # normal kill
                self.kill_piece(self.board_objects[self.field_idx])

        self.board_objects[self.field_idx] = self  # piece moves to field

        if not in_rochade:
            return mate_board
        else:
            return

    def kill_piece(self, enemy, replace=False):
        """kills (removes) piece from board or replaces it (in case of pawn promotion)"""

        if not replace:
            self.board_objects[enemy.field_idx] = None
        enemy.is_alive = False
        enemy.current_field = 'Out of field'
        enemy.field_idx = None
        return

    def resurrect_piece(self, enemy):
        """resurrects (adds) piece to board"""

        enemy.is_alive = True
        enemy.current_field = self.current_field
        enemy.field_idx = self.field_idx
        return

    def redo_move(self, mate_board, move_validity):
        """moves the piece back to previous position and status"""

        if move_validity == self.codes['opponent']:
            self.resurrect_piece(mate_board[self.field_idx])
        self.current_field = self.last_field
        self.field_idx = self.label2index(self.last_field)

    def reset_en_passant(self):
        """resets the possibility to perform en passant on pawns after move was executed"""

        for piece in GameOps.pieces:
            if not piece == self:
                piece.receive_en_passant = False
        return

    def pawn2queen(self):
        """promotes pawn to queen"""

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
        self.board_gui.add_piece(p.short_name, self.board_gui.images['pieces'][p.short_name], p.field_idx)
        mate_board = self.board_objects.copy()
        return mate_board

    def execute_move(self, move_to, is_test=False):
        """executes the move by performing the following operations:
            -) the first run of execute_move() is for the actually played move (is_test = False), the second run is to
                check for check and checkmate (is_test = True)
            1)  check whether move is valid
            2)  perform move if valid
            3)  if player would be in chess after move then revert back and output warning
            4)  -   if not in check and is_test = False then perform the move and check if opponent is checkmate
                -   if not in check and is_test = True (we are in the second run of execute_move() and are checking
                        for check and checkmate): all pieces go through their possible moves to see whether any move
                        can prevent checkmate. if no move can prevent check than this condition is never met and player
                        is checkmate
        """

        is_checkmate = True
        move_validity, valid_text = self.check_move_validity(move_to=move_to, color=self.color, is_test=is_test)
        if move_validity == self.codes['invalid']:
            if not is_test:
                self.output_text(valid_text, prefix='--', style='warning')
                return
            else:
                return is_checkmate

        mate_board = self.move_piece(move_to=move_to, move_validity=move_validity)
        if GameOps.is_rochade and not is_test:
            GameOps.rochade_rook.move_piece(
                move_to=GameOps.rochade_rook_move_to,
                move_validity=self.codes['empty'],
                in_rochade=True)
        if self.check_check(self.color):
            self.redo_move(mate_board=mate_board, move_validity=move_validity)
            for index, x in np.ndenumerate(self.board_objects):
                self.board_objects[index] = mate_board[index]
                if self.board_objects[index]:
                    self.board_objects[index].field_idx = index
                    self.board_objects[index].current_field = self.field_names[index]
            valid_text = 'CheckError: You would be in check position, try again!'
            GameOps.is_rochade_gui, GameOps.is_rochade = False, False
            if not is_test:
                self.board_gui.valid_move = False
                self.board_gui.kill = None
        else:
            if not is_test:
                self.output_text(valid_text)
                self.move_idx += 1
                self.move_counter()
                if GameOps.is_rochade:
                    GameOps.rochade_rook.move_idx += 1
                if self.check_pawn2queen():
                    self.pawn2queen()
                opp_color = 2 if self.color == 1 else 1
                self.check_checkmate(opp_color)
            else:
                is_checkmate = False
                self.redo_move(mate_board=mate_board, move_validity=move_validity)
                for index, x in np.ndenumerate(self.board_objects):
                    self.board_objects[index] = mate_board[index]

        if not is_test:
            # if valid_text:
            #     self.output_text(valid_text, prefix='--', style='warning')
            return
        else:
            return is_checkmate

    @classmethod
    def check_check(cls, color):
        """checks whether player is in check by checking if any of the opponent's piece attacks the king"""

        is_check = False
        for piece in GameOps:
            if isinstance(piece, King) and piece.color == color:
                move_to = piece.current_field  # find the King's position
                break

        is_test = True  # go into test mode so that piece are not actually moved
        for piece in GameOps:  # check if King is being attacked by any of opponent's pieces
            if piece.color != color and piece.is_alive:
                step = list(piece.get_move_step(move_to))
                piece.move_types(move_to, is_test=is_test)
                if step in piece.possible_moves:
                    move_validity, valid_text = piece.check_move_validity(
                        move_to=move_to, color=piece.color, is_test=is_test)
                    if not move_validity == cls.codes['invalid']:  # break if King is in check
                        is_check = True
                        break
        return is_check

    def check_checkmate(self, color):
        """checks whether player is in checkmate by checking if player can move out of check"""

        is_test = True
        if not self.check_check(color):  # player is not in check
            return

        self.output_text(
            text='{} needs to move out of check'.format('White' if color == 1 else 'Black'),
            prefix='--', style='warning')
        for piece in GameOps:
            if piece.color == color and piece.is_alive:
                for i, move_to in np.ndenumerate(self.field_names):
                    is_checkmate = piece.execute_move(move_to, is_test=is_test)
                    if not is_checkmate:
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

    def move(self, move_to):
        """governs the move"""

        move_to = move_to.upper()

        warning = ''
        if not self.move_count % 2 == (self.color - 1):
            warning = "TurnError: It is {}'s turn".format('white' if self.color == 2 else 'black')
        if not self.is_alive:
            warning = 'PieceError: {} was already killed'.format(self.short_name)
        if not np.array(np.where(self.field_names == move_to)).size:
            warning = 'FieldError: The Field does not exist! Try another value',

        if warning:
            self.output_text(text=warning, style='warning')
            return

        self.execute_move(move_to, is_test=False)
        self.reset_en_passant()
        return

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
        3) Pawn can onl kill in a one-step positive (relative to the black/white) diagonal direction or
            through en passant
    """

    normal_moves = [[1, 0]]
    special_moves = [[2, 0]]
    kill_moves = [[1, -1], [1, 1]]

    def move_types(self, move_to, is_test=False):
        if self.move_idx == 0:  # case 1)
            self.possible_moves = self.normal_moves + self.special_moves + self.kill_moves
        else:  # case 2) & 3)
            self.possible_moves = self.normal_moves + self.kill_moves
        if self.color == 2:  # black
            self.possible_moves = [[-i for i in m] for m in self.possible_moves]

        # exclude invalid moves
        kill_moves = self.possible_moves[-2:]
        move_vector = list(self.get_move_step(move_to))
        move_to_idx = self.label2index(move_to)
        move_valid = True

        self.perform_en_passant = False
        if abs(move_vector[0]) == 2 and not is_test:  # if pawn moves two in first move, it can receive en passant
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

    def move_types(self, move_to, is_test=False):
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

    def move_types(self, move_to, is_test=False):
        self.possible_moves = self.normal_moves
        return


class Bishop(GameOps):
    """Bishop can move diagonally"""

    def move_types(self, move_to, is_test=False):
        move_vector = list(self.get_move_step(move_to))
        if self.is_diagonal(move_vector):
            self.possible_moves = [move_vector]
        return


class Queen(GameOps):
    """Queen can move in any direction (diagonal, vertical, horizontal)"""

    def move_types(self, move_to, is_test=False):
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

    def move_types(self, move_to, is_test=False):
        self.possible_moves = self.normal_moves
        if self.move_idx == 0:
            move_vector = list(self.get_move_step(move_to))
            is_rochade, rook_rochade = self.check_rochade(move_vector, is_test=is_test)
            if is_rochade:
                self.possible_moves = self.normal_moves + [move_vector]
                self.set_rochade(move_vector, rook_rochade)
        return

    def check_rochade(self, move_vector, is_test=False):
        """checks whether the move is a valid rochade"""

        move_v, move_h = move_vector
        is_rochade, rook_rochade = False, None
        if move_vector in self.rochade and not GameOps.was_checked[self.color - 1] and not is_test:
            rooks = [p for p in GameOps if 'Rook' in p.name and p.color == self.color and p.is_alive]
            for rook in rooks:
                if (move_h > 0 and '2' in rook.name) or (move_h < 0 and '1' in rook.name) and rook.move_idx == 0:
                    is_rochade = True
                    rook_rochade = rook
                    break
        return is_rochade, rook_rochade

    def set_rochade(self, move_vector, rook_rochade):
        """sets all required settings for the rochade"""

        GameOps.is_rochade = True
        GameOps.is_rochade_gui = True
        GameOps.rochade_rook = rook_rochade
        rochade_idx = (self.field_idx[0], self.field_idx[1] + int(move_vector[1] / 2))
        GameOps.rochade_rook_move_to = self.index2label(rochade_idx)
        return
