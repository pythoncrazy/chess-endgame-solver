"""Lossless standard/Chess960 board snapshots and history-aware cache keys.

This module represents state and checks the context of draw claims. It does
not solve a FIDE tablebase of history states. A FEN alone cannot certify the
earlier repetition count.
"""
from collections import Counter
import hashlib
import struct

FIXED=struct.Struct('<4sBBQQB')


def uint(value):
    if not isinstance(value,int) or isinstance(value,bool) or value<0 or value.bit_length()>4096:
        raise ValueError('nonnegative integer with at most 4096 bits required')
    out=bytearray()
    while value>=128:out.append((value&127)|128);value>>=7
    out.append(value);return out


def take(raw,offset):
    value=shift=0
    while True:
        if offset>=len(raw) or shift>4096:raise ValueError('invalid integer encoding')
        byte=raw[offset];offset+=1;value|=(byte&127)<<shift
        if byte<128:
            if shift and byte==0:raise ValueError('noncanonical integer')
            if value.bit_length()>4096:raise ValueError('oversized integer')
            return value,offset
        shift+=7


def placement(board):
    out=bytearray(32)
    for square,piece in board.piece_map().items():
        code=piece.piece_type+(8 if piece.color else 0)
        out[square//2]|=code<<((square%2)*4)
    return bytes(out)


def snapshot(board):
    if board.uci_variant!='chess':raise ValueError('standard chess/Chess960 snapshot required')
    if board.fullmove_number<1 or board.promoted&~board.occupied:raise ValueError('invalid board metadata')
    ep=64 if board.ep_square is None else board.ep_square
    return FIXED.pack(b'CBS1',int(board.turn),int(board.chess960),board.castling_rights,
        board.promoted,ep)+placement(board)+uint(board.halfmove_clock)+uint(board.fullmove_number)


def restore(raw):
    import chess
    if len(raw)<FIXED.size+34:raise ValueError('truncated snapshot')
    magic,turn,variant,rights,promoted,ep=FIXED.unpack_from(raw)
    if magic!=b'CBS1' or turn>1 or variant>1 or ep>64:raise ValueError('invalid snapshot')
    board=chess.Board(None,chess960=bool(variant));pieces=raw[FIXED.size:FIXED.size+32]
    for square in range(64):
        code=(pieces[square//2]>>((square%2)*4))&15
        if code in (7,8,15):raise ValueError('invalid piece code')
        if code:board.set_piece_at(square,chess.Piece(code&7,bool(code&8)),promoted=bool(promoted&(1<<square)))
    clock,offset=take(raw,FIXED.size+32);fullmove,offset=take(raw,offset)
    if offset!=len(raw) or fullmove<1 or promoted&~board.occupied:raise ValueError('invalid snapshot tail')
    board.turn=bool(turn);board.castling_rights=rights;board.ep_square=None if ep==64 else ep
    board.halfmove_clock=clock;board.fullmove_number=fullmove
    return board


def repetition_token(board):
    # Only a legal en-passant possibility changes standard repetition identity.
    ep=64 if not board.has_legal_en_passant() else board.ep_square
    return placement(board)+struct.pack('<BQB',int(board.turn),board.clean_castling_rights(),ep)


def repetition_counts(board):
    history=board.copy(stack=True);counts=Counter([repetition_token(history)])
    while history.move_stack:
        move=history.pop()
        if history.is_irreversible(move):break
        counts[repetition_token(history)]+=1
    return counts


def can_claim_threefold(board,counts=None):
    counts=repetition_counts(board) if counts is None else counts
    if counts[repetition_token(board)]>=3:return True
    child=board.copy(stack=False)
    for move in list(child.legal_moves):
        child.push(move)
        repeated=counts[repetition_token(child)]>=2
        child.pop()
        if repeated:return True
    return False


def cache_identity(board,metric,rule_version,history_complete=False,history_aware=True):
    if not isinstance(history_complete,bool) or not isinstance(history_aware,bool):raise ValueError('explicit boolean history flags required')
    if not isinstance(metric,str) or not isinstance(rule_version,str) or not metric or not rule_version:
        raise ValueError('metric and rule version required')
    if history_aware and not history_complete:raise ValueError('complete prior history required for history-aware identity')
    canonical=board.copy(stack=True);canonical.fullmove_number=1
    raw=bytearray(snapshot(canonical))  # Move ordinal does not change game value.
    for label in (metric,rule_version):
        encoded=label.encode();raw.extend(uint(len(encoded)));raw.extend(encoded)
    raw.append(int(history_aware))
    if history_aware:
        counts=repetition_counts(board)
        for key,count in sorted(counts.items()):raw.extend(key);raw.extend(uint(count))
    return hashlib.sha256(raw).hexdigest()


def game_snapshot(board,history_complete=False):
    if not isinstance(history_complete,bool):raise ValueError('explicit history completeness required')
    raw=snapshot(board);counts=repetition_counts(board)
    out=bytearray(b'CGS1'+uint(len(raw))+raw+bytes([int(history_complete)])+uint(len(counts)))
    for key,count in sorted(counts.items()):out.extend(key);out.extend(uint(count))
    return bytes(out)


def restore_game(raw):
    if raw[:4]!=b'CGS1':raise ValueError('unknown game snapshot')
    length,offset=take(raw,4);end=offset+length
    if end>=len(raw):raise ValueError('truncated game snapshot')
    board=restore(raw[offset:end]);known=raw[end]
    if known>1:raise ValueError('invalid history flag')
    count,offset=take(raw,end+1)
    if count>(len(raw)-offset)//43:raise ValueError('invalid repetition count')
    counts=Counter();previous=None
    for _ in range(count):
        key=raw[offset:offset+42];offset+=42
        if len(key)!=42 or (previous is not None and key<=previous):raise ValueError('invalid repetition order')
        occurrences,offset=take(raw,offset)
        if not occurrences:raise ValueError('empty repetition entry')
        counts[key]=occurrences;previous=key
    if offset!=len(raw) or counts[repetition_token(board)]<1:raise ValueError('invalid repetition tail/current state')
    return board,counts,bool(known)
