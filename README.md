# Extra Layers and helpers for the Matlab Deep Learning and Reinforcement Learning Toolkits

## onehotLayer

Converts integer index values into "one hot" encoded (i.e. indicator) variables within the network so you don't have to do it in the data.  Particularly helpful for discrete action spaces when not using the `rlVectorQValueFunction`.

## PositionEncode

The "Attention is all you need" fixed position encoding scheme.  Slightly modified to support non-contigous positions.

## extractTime

Extract one time from a sequential input, like LSTM layer OutputMode='last', but without the LSTM part.  Also modified to allow you select a particular item in the sequence.
