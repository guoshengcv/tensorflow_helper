import re


__all__ = [
  'in_res'
]


def in_res(string, regexs):
  for regex in regexs:
    if regex.fullmatch(string) != None:
      return True
  return False
