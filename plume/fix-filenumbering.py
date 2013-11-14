#!/usr/bin/env python

import argparse
import os
import os.path
import tables
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filenames', nargs='*', type=str, help='Files to fix.')
    args = parser.parse_args()

    for filename in args.filenames:
        with tables.open_file(filename) as fileh:
            repeat = fileh.root.repeat[0]
        components = filename.split('.')
        components[-2] = str(repeat)
        fixed_filename = ''.join(components)

        if os.path.exists(fixed_filename):
            sys.stderr.write(
                '{} already exists.{}'.format(fixed_filename, os.linesep))
        else:
            os.rename(filename, fixed_filename)
