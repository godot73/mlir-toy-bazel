"""
Example input:
463:    %61 = linalg.matmul ins(%collapsed, %58 : tensor<30x768xf32>, tensor<768x768xf32>) outs(%60 : tensor<30x768xf32>) -> tensor<30x768xf32>
"""

import collections
import re
from typing import Iterable

SHAPE_PATTERN = r'tensor\<(\d+)x(\d+)xf32\>'

PATTERN = re.compile('.*'.join([SHAPE_PATTERN, SHAPE_PATTERN, SHAPE_PATTERN, SHAPE_PATTERN]))

def extract(matches: Iterable[int]):
    # e.g. 30, 3072, 3072, 768, 30, 768, 30, 768
    assert len(matches) == 8
    row, inner, _, col, _, _, _, _ = matches
    assert row == matches[4] == matches[6]
    assert inner == matches[2]
    assert col == matches[5] == matches[7]
    return row, inner, col

def gen_dims():
    with open('/tmp/linalg_matmul.txt', 'r') as infile:
        for line in infile:
            match = PATTERN.search(line)
            if match:
                matches = [int(match.group(i + 1)) for i in range(8)]
                dims = extract(matches)
                print('dims = {}'.format(dims))
                yield dims

def main():
    count = collections.Counter(gen_dims())
    print(count)


if __name__ == '__main__':
    main()

