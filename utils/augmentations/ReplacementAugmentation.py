import torch.nn as nn
import random

class ReplaceCharacterAugmentation(nn.Module):
    def __init__(self, prob=0.2):
        super().__init__()
        self.prob = prob
        self.replacement_map = {
            # Latin letters and similar looking digits/symbols
            'a': ['@', 'á', 'à', 'â', 'ä', 'ɑ'],  # a variants + alpha-like
            'b': ['6', 'ß', 'Β'],  # beta-like
            'c': ['ç', '¢', 'ć', 'č'],
            'd': ['đ', 'ð', 'cl'],  # ð Icelandic, 'cl' for OCR confusion
            'e': ['3', 'é', 'è', 'ê', 'ë', '£'],  
            'f': ['ƒ'],
            'g': ['9', 'q', 'ɡ'],
            'h': ['lh', 'н'],  # Cyrillic en
            'i': ['1', '!', 'í', 'ì', 'î', 'ï', 'ı'],  
            'j': ['ј'],  # Cyrillic je
            'k': ['κ', 'lk'],
            'l': ['1', '!', 'ł'],
            'm': ['nn', 'rn', 'ṃ'],
            'n': ['m', 'ń', 'ñ'],
            'o': ['0', 'ó', 'ò', 'ô', 'ö', 'ø', 'ơ', 'ọ', 'о'],  # Cyrillic o
            'p': ['ρ', 'р'],  # Greek rho + Cyrillic er
            'q': ['9', 'g'],
            'r': ['ɼ', 'ʀ'],
            's': ['5', '$', 'š', 'ś'],
            't': ['7', '+', 'ţ', 'ť'],
            'u': ['ü', 'ù', 'ú', 'û', 'ư'],
            'v': ['ⅴ', 'ν'],  # Roman numeral five + Greek nu
            'w': ['vv', 'ш'],
            'x': ['×', '%', 'χ'],
            'y': ['ý', 'ỳ', '¥'],
            'z': ['2', 'ž', 'ź', 'ż'],
            # Uppercase variants
            'A': ['4', '@', 'Á', 'À', 'Â', 'Ä', 'Λ'],
            'B': ['8', 'ß', 'Β'],
            'C': ['Ç', 'Ć', 'Č'],
            'D': ['Ð', 'Đ'],
            'E': ['3', 'É', 'È', 'Ê', 'Ë', '£'],
            'F': ['Ƒ'],
            'G': ['6', '9'],
            'H': ['Н'],
            'I': ['1', '!', 'Í', 'Ì', 'Î', 'Ï', 'Ι'],
            'J': ['Ј'],
            'K': ['Κ'],
            'L': ['1', '!', 'Ł'],
            'M': ['NN', 'Μ'],
            'N': ['Ń', 'Ñ'],
            'O': ['0', 'Ó', 'Ò', 'Ô', 'Ö', 'Ø', 'Ơ', 'Ọ', 'О'],
            'P': ['Ρ', 'Р'],
            'Q': ['9'],
            'R': ['Ɍ', 'ʀ'],
            'S': ['5', '$', 'Š', 'Ś'],
            'T': ['7', '+', 'Ţ', 'Ť'],
            'U': ['Ü', 'Ù', 'Ú', 'Û', 'Ư'],
            'V': ['Ⅴ', 'Ν'],
            'W': ['VV', 'Ш'],
            'X': ['×', '%', 'Χ'],
            'Y': ['Ý', 'Ỳ', '¥'],
            'Z': ['2', 'Ž', 'Ź', 'Ż'],
            # Digits (common OCR confusions)
            '0': ['O', 'o', 'Ø', 'Ө'],
            '1': ['I', 'l', 'í', '!', 'ı'],
            '2': ['Z', 'z'],
            '3': ['E', 'e'],
            '5': ['S', 's', '$'],
            '6': ['b', 'G', 'g'],
            '7': ['T', 't'],
            '8': ['B', 'ß'],
            '9': ['g', 'q', 'p'],
            # Some symbols
            '@': ['a', 'A'],
            '$': ['S', 's'],
            '+': ['t', 'T'],
            '%': ['x', 'X'],
        }

    def forward(self, x: str) -> str:
        out = []
        for c in x:
            if c in self.replacement_map and random.random() < self.prob:
                out.append(random.choice(self.replacement_map[c]))
            else:
                out.append(c)
        return ''.join(out)

    def __call__(self, sample):
        return self.forward(sample)