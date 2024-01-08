#!/usr/bin/env python3
"""
A script for opening a file with a web browser from vim.

Add the following lines to your .vimrc and add a symlink to this script from a
directory in $PATH.

nmap \cf :exe ":silent !sccs.py --service=github --file=" . expand("%:~:.") <CR>
nmap \cw :exe ":silent !sccs.py --service=github --file=" . expand("%:~:.") . ' --word=' . expand("<cword>") <CR>
nmap \cl :exe ":silent !sccs.py --service=github --file=" . expand("%:~:.") . ' --line=' . line(".") <CR>

From vim, \cf, \cw, \cl invokes this script with --file and optional --word
and --line arguments. Then, this script opens, for example, a github page using
google-chrome.

To use this script with opengrok, start opengrok services from a server and set
its host name (e.g. localhost) as $OPENGROK environment variable.
"""

import argparse
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Optional, Tuple
import urllib

# This script assumes all repositories are direct children of a directory named
# 'repos'.
_REPOSITORY_ROOT = 'repos'
_SERVICE_NAME_GITHUB = 'github'
_SERVICE_NAME_OPENGROK = 'opengrok'

# Map from a known github repo owner to their projects and default branches.
_GITHUB_PROJECT_AND_BRANCH_BY_OWNER = {
    'godot73': [('assets', 'master')],
    'openxla': [('iree', 'main')],
    'nod-ai': [('iree-amd-aie', 'main'), ('SHARK', 'main'),
               ('SHARK-Turbine', 'main'), ('SRT', 'shark')],
    'llvm': [('torch-mlir', 'main'), ('llvm-project', 'main')],
}

# Map from a known project to its owner and the default branch.
# The contents look like: {'iree': ('openxla', 'main'), ...}
_GITHUB_OWNER_AND_BRANCH_BY_PROJECT = {}
for owner, projects in _GITHUB_PROJECT_AND_BRANCH_BY_OWNER.items():
    for project, branch in projects:
        _GITHUB_OWNER_AND_BRANCH_BY_PROJECT[project] = (owner, branch)

# Map from a known project to its port number at the opengrok server.
_OPENGROK_PORT_BY_PROJECT = {
    'llvm-project': 9090,
    'iree': 9091,
    'SRT': 9091,
    'torch-mlir': 9093,
    'pytorch': 9094,
}


def open_github(project: str, subtree: str, line: Optional[int],
                keyword: Optional[str]) -> None:
    owner, branch = _GITHUB_OWNER_AND_BRANCH_BY_PROJECT[project]
    domain = 'http://github.com'
    if keyword:
        encoded_query = urllib.parse.quote(f'repo:{owner}/{project} {keyword}',
                                           safe='')
        url = f'{domain}/search?q={encoded_query}&type=code'
    else:
        url = f'{domain}/{owner}/{project}/blob/{branch}/{subtree}'
        if line:
            url += f'#L{line}'
    command = ['google-chrome', url]
    print('command={}'.format(command))
    subprocess.run(command)


def open_opengrok(project: str, subtree: str, line: Optional[int],
                  keyword: Optional[str]) -> None:
    port = _OPENGROK_PORT_BY_PROJECT[project]
    domain = os.environ['OPENGROK']  # e.g. 'localhost'
    assert domain is not None
    host = f'http://{domain}:{port}'
    if keyword:
        url = (
            '{}/search?full={}&defs=&refs=&path=&hist=&type=&xrd='
            '&si=full&searchall=true&si=full'
        ).format(host, keyword)
    else:
        url = f'{host}/xref/{subtree}'
        if line:
            url += f'#{line}'
    command = ['google-chrome', url]
    print('command={}'.format(command))
    subprocess.run(command)


# Extracts out the repository name and the subpath from args.file.
# Assumes the file name looks like '.../repos/[repo name]/.../[filename]'.
# In other words, the repository is a child of a directory named `repos`.
# Example:
#   extract_subdir('~/repos/my_project/a/b/c/d.txt')
#   == ('my_project', 'a/b/c/d.txt')
def extract_repo_subtree(filename: str) -> Tuple[str, str]:
    assert filename is not None
    filepath = pathlib.Path(args.file).expanduser().resolve().absolute()
    print('filepath = {}'.format(filepath))
    root_loc = filepath.parts.index(_REPOSITORY_ROOT)
    return (filepath.parts[root_loc + 1],
            '/'.join(filepath.parts[root_loc + 2:]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        help='The name of the file to open',
                        default=None)
    parser.add_argument('--service',
                        help='The name of the code search service; '
                        'currently supports github and opengrok.',
                        default=_SERVICE_NAME_GITHUB)
    parser.add_argument('--line',
                        help='The line number',
                        type=int,
                        default=None)
    parser.add_argument('--word',
                        help='The keyword to search',
                        type=str,
                        default=None)
    args = parser.parse_args()
    print('args={}'.format(args))
    return args


if __name__ == '__main__':
    args = parse_args()
    print('repo name = {}'.format(extract_repo_subtree(args.file)))
    project, subtree = extract_repo_subtree(args.file)
    if args.service == _SERVICE_NAME_GITHUB:
        open_github(project, subtree, args.line, args.word)
    else:
        args.service == _SERVICE_NAME_OPENGROK
        open_opengrok(project, subtree, args.line, args.word)
