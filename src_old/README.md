## Uporaba

`net = Network([10,5,4,2])`

Ta mreža predstavlja 10 inputov in 2 ouputa.

Prvi pristop je čisto preveč matematično "neumen". Je predstavljivo vse kar se more zgoditi, vendar bi moral na roke implementirati množenje matrik, kar pa seveda ni najbolj pametno (in je v pytohonu čisto preveč počasno!!).

## Beware

### Stari approach

Zaenkrat ni implementiran dropout in backpropagation.
Prvi approach bo zelo inefficient, ker bom racunal odvod za vsak nevron (connection) posebej.
