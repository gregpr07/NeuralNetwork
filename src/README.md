# src

## Mreža

Glavna koda je vsebovana v `network.py`, ki vsebuje skor vse razred in skoraj vse metode mreže.

## Kernel

Zaenkrat so za uporabo implementirani 3 kerneli, vendar jih lahko dodam poljubno veliko.

Ko dodamo nov layer moremo napisati tudi kateri kernel bo imel nevron in je tretnutno eden izmed:
`['ReLu', 'Sigmoid', 'Softmax']`

## Cost

Na voljo je samo Mean-Squared kernel, vendar bi lahko dodal tudi novega (recimo cross-entropy) če dodam novo funkcijo.
