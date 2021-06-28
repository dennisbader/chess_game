# Chess Game

My first developed game from 2018. 
So far the game is supposed to be played with both sides.
Playing against the computer might be added in a future version.

## Installation

Clone the repository to your local machine.
> git clone https://github.com/dennisbader/chess_game.git

Requirements (tested):
* python=3.7
* numpy=1.17.2
* matplotlib=3.1.1
* tk=8.6.8

For an easy conda environment installation you can find a .yml file at ./env_builder/chess_gui_env.yml

## How to play

* You can run the game by executing the `chess_game.py` script:
> python your_path_to_chess_game/chess_game.py

* To move a piece: first click on the piece, then click on the field you want to move to.
* Undo/redo move: you can always go back and forth between the move history by clicking the UNDO/REDO buttons
* The game should only allow valid moves. In case your move is invalid, the command-line interface interface will tell why it is invalid.
* Have fun playing!
