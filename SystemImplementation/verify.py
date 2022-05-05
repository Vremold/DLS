import sys
import os
import json

from System import UI
from ..pathutil import VERIFICATION_DATA_DIR
ui = UI()

def test(alpha=0.35):
    success = 0
    mrr = 0
    with open(VERIFICATION_DATA_DIR, "r", encoding="utf-8") as inf:
        for line in inf:
            line = line.strip()
            text, ans = line.split("\t")
            _, _, repos = ui.query(text)
            repos = [repo for repo, _ in repos[:20]]
            if ans in repos:
                success += 1
                mrr += 1 / (repos.index(ans) + 1)
        print("Alpha[{}]: {}\t{}\n".format(alpha, success, mrr))

alpha = 0.35
while alpha <= 1:
    if alpha == 0.35:
        alpha += 0.05
        continue
    test(alpha)
    alpha += 0.05
test(0.35)
