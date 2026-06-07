import json
from dataclasses import dataclass, is_dataclass, asdict


@dataclass(slots=True)
class A:
    a: int = 1
    b: str = "hi"


x = A(a=4)
y = A(a=5, b="hello")

d = dict(x=x, y=y)
to_dump = {"meta": d, "lol!": [1, 2, 3, 4]}


class DataclassDictConverterEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def DataclassDictConverterDecoder(obj):
    if obj.keys() == A.__annotations__.keys():
        return A(**obj)
    return obj


with open("test.json", "w") as file:
    json.dump(to_dump, file, cls=DataclassDictConverterEncoder)

with open("test.json", "r") as file:
    e = json.load(file, object_hook=DataclassDictConverterDecoder)

print(e)
assert to_dump == e, print(f"{d=}", f"{e=}")
