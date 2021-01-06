# NeuralNetwork

Šolski projekt v katerem bom naredil svojo nevronsko mrežo brez uporabe že narejenih knjižnic (razen `Numpy` za hitrejše množenje matrik).

## Opis

Vsa koda za nevronsko mrežo, ki jo uporablja `demo.ipynb` je v `./src`.

## Instalacija

`pip install -r requirements.txt`

(Vbistvu je za dejanski network potrebno naložiti samo `Numpy` zato je bolj relevanten za `demo.ipynb`, kjer sem uporabljal keras, da sem naložil MNIST databazo.)

## Uporaba

`net = Network()`

Da pokličemo Network.

`net.addInputLayer(dim_0)`

Da dodamo nov input layer (to je potrebno preden kličemo `.addLayer()`).

`net.addLayer(dim_L,<kernel>)`

Da dodamo celotno povezan hidden (ali zadnji) layer.
`<kernel>` je eden izmed kernelov. Ko kličemo funkcijo dobimo kateri kerneli so na voljo - več podatkov v `README.md` v `./src`.

`net.train(X,Y)`

kjer sta `X,Y` vhodna in izhodna vektorja (iz databaze na kateri treniramo).
`y.shape[1]` mora biti enaka `dim_0`.
`x.shape[1]` mora biti enaka `dim_L`.

`net.predict(x)`

S tem pridemo do dejanskega predloga mreže (vizualno prikazano v demotu).

## Uspeh

Nauči se:

### mnist_784 databaza od scikit-learn

90% accuracy po 10 epochih.

## Beware

Ni implementiranega dropout-a tako, da zna priti do overfita.

## Viri in literatura

- https://www.youtube.com/watch?v=tIeHLnjs5U8
- https://en.wikipedia.org/wiki/Backpropagation
- https://www.guru99.com/backpropogation-neural-network.html
- https://medium.com/@udaybhaskarpaila/multilayered-neural-network-from-scratch-using-python-c0719a646855
- https://mlfromscratch.com/neural-network-tutorial/#/ (accuracy function)
