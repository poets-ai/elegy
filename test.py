import typing as tp

DictLike = tp.Union[tp.Dict, tp.List[tp.Tuple]]

d = {}

if isinstance(d, DictLike):
    print("yes")
else:
    print("no")
