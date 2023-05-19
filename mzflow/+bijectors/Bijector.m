classdef Bijector < nnet.layer.Layer ...
        & nnet.layer.Formattable
    methods (Abstract)
        inverse(this, inputs, conditions);
        info(this);
    end
end
