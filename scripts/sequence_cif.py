"""Read sequence categories without retaining coordinate tables.

Uses Biopython's pinned mmCIF tokenizer to preserve quoted/multiline values.
Scans the entire gzip stream, so category order does not affect correctness.
"""
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def read_categories(handle, prefixes, atom_filter=None):
    lexer = MMCIF2Dict.__new__(MMCIF2Dict)
    lexer.quote_chars = ["'", '"']
    lexer.whitespace_chars = [' ', '\t']
    tokens = iter(lexer._tokenize(handle))
    first = next(tokens, '')
    if not first.lower().startswith('data_'):
        raise ValueError('mmCIF data block missing')
    keep = lambda key: key.startswith(prefixes)
    output = {}
    pending = None
    while True:
        token = pending if pending is not None else next(tokens, None)
        pending = None
        if token is None:
            break
        if token.lower() == 'loop_':
            keys = []
            value = next(tokens, None)
            while value is not None and value.startswith('_'):
                keys.append(value)
                if keep(value):
                    output[value] = []
                value = next(tokens, None)
            if not keys:
                raise ValueError('empty loop')
            column = 0
            atom_loop = any(k.startswith('_atom_site.') for k in keys)
            retained = [(j, k) for j, k in enumerate(keys) if keep(k)]
            row = []
            while value is not None:
                if column == 0 and (value.startswith('_') or value.lower() in ('loop_', 'stop_') or value.lower().startswith('data_')):
                    pending = value
                    break
                if retained:
                    row.append(value)
                column = (column + 1) % len(keys)
                if column == 0 and retained:
                    if not atom_loop or atom_filter is None or atom_filter(dict(zip(keys, row))):
                        for j, key in retained:
                            output[key].append(row[j])
                    row = []
                value = next(tokens, None)
            if column:
                raise ValueError('incomplete loop row')
        elif token.startswith('_'):
            value = next(tokens, None)
            if value is None:
                raise ValueError('missing scalar value')
            if keep(token):
                output[token] = [value]
    return output


def read_sequence_categories(handle):
    return read_categories(handle, ('_entry.', '_entity_poly.', '_entity_poly_seq.'))
