import os
import sys
import h5py
import numpy
import random
import chess
import chess.pgn
import itertools
import multiprocessing
from joblib import Parallel, delayed


def get_games(infile):
    """
    Open game files and use chess.pgn module to
    read games properly
    """
    f = open(infile)

    while True:
        try:
            game = chess.pgn.read_game(f)
        except KeyboardInterrupt:
            raise
        except:
            continue

        if not game:
            break
        
        yield game


def bb2array(b, flip=False):
    """
    Game board to array
    """
    x = numpy.zeros(64, dtype=numpy.int8)
    
    for pos, piece in enumerate(b.pieces):
        if piece != 0:
            color = int(bool(b.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
            col = int(pos % 8)
            row = int(pos / 8)
            if flip:
                row = 7-row
                color = 1 - color

            piece = color*7 + piece

            x[row * 8 + col] = piece

    return x


def process_game(g):
    """
    Process the game, extracting most valuable info
    """
    rm = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
    r = g.headers['Result']
    if r not in rm:
        return None
    y = rm[r]

    # Generate all boards
    gn = g.end()
    if not gn.board().is_game_over():
        return None

    gns = []
    moves_left = 0
    while gn:
        gns.append((moves_left, gn, gn.board().turn == 0))
        gn = gn.parent
        moves_left += 1

    print(len(gns))
    if len(gns) < 10:
        print(g.end())

    gns.pop()

    moves_left, gn, flip = random.choice(gns) # remove first position

    b = gn.board()
    x = bb2array(b, flip=flip)
    b_parent = gn.parent.board()
    x_parent = bb2array(b_parent, flip=(not flip))
    if flip:
        y = -y

    # generate a random baord
    moves = list(b_parent.legal_moves)
    move = random.choice(moves)
    b_parent.push(move)
    x_random = bb2array(b_parent, flip=flip)

    if moves_left < 3:
        print("Moves Left:")
        print(moves_left) 
        print("Winner:")
        print(y)
        print(g.headers)
        print(b)
        print("Checkmate:")
        print(g.end().board().is_checkmate())

    return (x, x_parent, x_random, moves_left, y)


def read_all_games(infile, outfile):
    """
    Read in the games inside directory and
    create hdf5 format files to store dataset info
    """
    hfile = h5py.File(outfile, 'w')
    X, Xr, Xp = [hfile.create_dataset(d, (0, 64), dtype='b', maxshape=(None, 64), chunks=True) for d in ['x', 'xr', 'xp']]
    Y, M = [hfile.create_dataset(d, (0,), dtype='b', maxshape=(None,), chunks=True) for d in ['y', 'm']]
    size = 0
    line = 0
    for game in get_games(infile):
        game = process_game(game)
        if game is None:
            continue
        x, x_parent, x_random, moves_left, y = game

        if line + 1 >= size:
            hfile.flush()
            size = 2 * size + 1
            print("Resizing to:")
            print(size)
            [d.resize(size=size, axis=0) for d in (X, Xr, Xp, Y, M)]

        X[line] = x
        Xr[line] = x_random
        Xp[line] = x_parent
        Y[line] = y
        M[line] = moves_left

        line += 1

    [d.resize(size=line, axis=0) for d in (X, Xr, Xp, Y, M)]

    hfile.close()

def read_all_games_2(a):
    return read_all_games(*a)


def get_files():
    """
    Read in files and pass to processing pool to 
    convert format and store
    """
    files = []
    data_path = 'Data/800-999'
    for filename in os.listdir(data_path):
        if not filename.endswith('.pgn'):
            continue
        infile = os.path.join(data_path, filename)
        outfile = infile.replace('.pgn', '.hdf5')
        if not os.path.exists(outfile):
            files.append((infile, outfile))
    print(files)
    pool = multiprocessing.Pool()
    pool.map(read_all_games_2, files)


if __name__ == '__main__':
    get_files()