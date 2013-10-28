#!/usr/bin/env python

import argparse
import tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filename', nargs='*', type=str, help='File to check')
    args = parser.parse_args()

    for filename in args.filename:
        with tables.open_file(filename) as fileh:
            if 'exception' in fileh.root:
                error = fileh.root.exception[0]
                print error.__class__.__name__, filename
