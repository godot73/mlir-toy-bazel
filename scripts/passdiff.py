#!/usr/bin/env python3
"""A script for diffing between IRs before and after passes.

Prerequisite 0: Symlink to this file from somehwer in $PATH.

Prerequisite 1: Generate a compilation log first by running:
    - iree-compile [.mlir] -o [output.vmfb] \
            --mlir-print-ir-before-all \
            --mlir-print-ir-after-all \
            --mlir-elide-elementsattrs-if-larger=32 \
            [other flags] > ~/log/output.log

Usage 1: To generate diffs, run:
    passdiff.py --mode=generate [--dryrun] --input=~/log/output.log

Usage 2: To view diff, change to a diff directory and run:
    passidff.py --input=[prefix to a diff pass dir; e.g. 000]
"""

import argparse
import collections
import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Optional, Tuple

DIFFTOOL = 'tkdiff'
DIFFDIR = '~/passdiff'

_MODE_GENERATE = 'generate'
_MODE_VIEWDIFF = 'viewdiff'

_DUMP_BEFORE_PATTERN = re.compile(r'IR Dump Before .*\((.*)\)')
_DUMP_AFTER_PATTERN = re.compile(r'IR Dump After .*\((.*)\)')

_BEFORE_BLOCK = 'before'
_AFTER_BLOCK = 'after'

_BEFORE_FILENAME = 'before.mlir'
_AFTER_FILENAME = 'after.mlir'


class PassList(object):

    def __init__(self):
        self.count_by_passname = collections.defaultdict(int)
        # Storage to keep the order of passes. Tuple of passname and its count.
        self.passes = []

    def add_passname(self, passname: str) -> Tuple[str, int]:
        count = self.count_by_passname[passname]
        self.passes.append((passname, count))
        self.count_by_passname[passname] = count + 1
        return (passname, count)

    def get_passname(self, passname: str) -> Tuple[str, int]:
        count = self.count_by_passname.get(passname)
        assert count is not None
        return (passname, count - 1)

    def __str__(self):
        return 'PassList(passes={}, count_by_passname={})'.format(
            self.passes, self.count_by_passname)


class BlockGenerator(object):

    def __init__(self, args):
        self.args = args
        self.passlist = PassList()
        self.save_root = Path.absolute(Path.expanduser(
            Path(DIFFDIR))).joinpath(
                datetime.datetime.today().strftime('%Y%m%d-%H%M%S'))
        # Indexed by (passname, "before" or "after")
        self.lines_by_pass = collections.defaultdict(list)

    def gen_lines(self) -> Iterable[str]:
        with open(Path.absolute(Path.expanduser(Path(self.args.input))),
                  'r') as infile:
            for line in infile:
                stripped = line.rstrip()
                yield stripped

    def generate(self) -> None:
        passname = None
        for linenum, line in enumerate(self.gen_lines()):
            maybe_passname = self.maybe_before_passname(line)
            if maybe_passname is not None:
                passname = maybe_passname
                current_block = _BEFORE_BLOCK
                continue
            maybe_passname = self.maybe_after_passname(line)
            if maybe_passname is not None:
                passname = maybe_passname
                current_block = _AFTER_BLOCK
                continue
            if passname is None:
                continue
            self.lines_by_pass[(passname, current_block)].append(line)

    def maybe_before_passname(self, line: str) -> Optional[str]:
        matches = _DUMP_BEFORE_PATTERN.search(line)
        if matches:
            raw_passname = matches.group(1)
            return self.passlist.add_passname(raw_passname)
        return None

    def maybe_after_passname(self, line: str) -> Optional[str]:
        matches = _DUMP_AFTER_PATTERN.search(line)
        if matches:
            raw_passname = matches.group(1)
            return self.passlist.get_passname(raw_passname)
        return None

    def save_blocks(self) -> None:
        for index, pass_and_count in enumerate(self.passlist.passes):
            passname, count = pass_and_count
            before_block = self.lines_by_pass.get(
                (pass_and_count, _BEFORE_BLOCK))
            assert before_block is not None
            after_block = self.lines_by_pass.get(
                (pass_and_count, _AFTER_BLOCK))
            if after_block is None:
                print(
                    'Pass {} has no after-pass block.'.format(pass_and_count))
                continue
            if before_block == after_block:
                print('Pass {} has no diff.'.format(pass_and_count))
                continue
            self.save_block(index, passname, before_block, after_block)

    def save_block(self, index: int, passname: str, before_block: str,
                   after_block: str) -> None:
        path_dir = self.save_root / ('{:03d}-{:s}'.format(index, passname))
        print('Saving at {}'.format(path_dir))
        if not self.args.dryrun:
            Path.mkdir(path_dir, parents=True)
            self.save_file(path_dir / _BEFORE_FILENAME, before_block)
            self.save_file(path_dir / _AFTER_FILENAME, after_block)

    def save_file(self, fullpath: Path, lines: Iterable[str]) -> None:
        with open(fullpath, 'w') as outfile:
            for line in lines:
                outfile.write(f'{line}\n')


def viewdiff(dir_prefix: str) -> None:
    diffdir = find_diffdir(dir_prefix)
    if diffdir is None:
        print('Cannot find a subdirectory starting with {}'.format(dir_prefix),
              file=sys.stderr)
        exit(1)
    print('Selected dir = {}'.format(diffdir))
    diffpath = Path.cwd() / diffdir
    rundiff(diffpath / _BEFORE_FILENAME, diffpath / _AFTER_FILENAME)


def find_diffdir(dir_prefix: str) -> Optional[str]:
    assert dir_prefix
    dirs = sorted(os.listdir(Path.cwd()))
    for d in dirs:
        if d.startswith(dir_prefix):
            return d
    return None


def rundiff(before_filepath: str, after_filepath: str):
    subprocess.run([DIFFTOOL, before_filepath, after_filepath])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        help='Either generate or viewdiff (default)',
                        type=str,
                        choices=[_MODE_GENERATE, _MODE_VIEWDIFF],
                        default=_MODE_VIEWDIFF)
    parser.add_argument(
        '--input',
        help=('With --mode=generate, input log file containing IRs '
              'before and after each pass, '
              'or --mode=viewdiff, prefix of a pass directory name'),
        type=str,
        required=True)
    parser.add_argument('--dryrun',
                        help=('If set with --mode=generate, '
                              'it runs everything but saving the diff files.'),
                        action=argparse.BooleanOptionalAction,
                        default=False)
    args = parser.parse_args()
    print('args={}'.format(type(args)))
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.mode == _MODE_GENERATE:
        generator = BlockGenerator(args)
        generator.generate()
        generator.save_blocks()
    else:
        assert args.mode == _MODE_VIEWDIFF
        viewdiff(args.input)
