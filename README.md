# NeuralNetwork

Šolski projekt v katerem bom naredil svojo nevronsko mrežo brez uporabe že narejenih knjižnic.

## Uporaba

`net = Network([10,5,4,2])`

Ta mreža predstavlja 10 inputov in 2 ouputa.

Prvi pristop je čisto preveč matematično "neumen". Je predstavljivo vse kar se more zgoditi, vendar bi moral na roke implementirati množenje matrik, kar pa seveda ni najbolj pametno (in je v pytohonu čisto preveč počasno!!).

## Beware

### Stari approach

Zaenkrat ni implementiran dropout in backpropagation.
Prvi approach bo zelo inefficient, ker bom racunal odvod za vsak nevron (connection) posebej.

## Viri in literatura

- https://www.youtube.com/watch?v=tIeHLnjs5U8
- https://en.wikipedia.org/wiki/Backpropagation
- https://www.guru99.com/backpropogation-neural-network.html
- https://medium.com/@udaybhaskarpaila/multilayered-neural-network-from-scratch-using-python-c0719a646855
