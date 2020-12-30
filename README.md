# NeuralNetwork

Šolski projekt v katerem bom naredil svojo nevronsko mrežo brez uporabe že narejenih knjižnic.

## Uporaba

`net = Network()`

`net.addInputLayer(dim_0)`

`net.addLayer(dim_L)`

`net.train(X,Y)`

where X and Y are batches of input and output vectors that match the sizes of first and last layer.

## Beware

Ni implementiranega dropout-a tako, da zna priti do overfita.

## Viri in literatura

- https://www.youtube.com/watch?v=tIeHLnjs5U8
- https://en.wikipedia.org/wiki/Backpropagation
- https://www.guru99.com/backpropogation-neural-network.html
- https://medium.com/@udaybhaskarpaila/multilayered-neural-network-from-scratch-using-python-c0719a646855
- https://mlfromscratch.com/neural-network-tutorial/#/ (accuracy function)
