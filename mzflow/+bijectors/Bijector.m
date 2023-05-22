classdef Bijector < nnet.layer.Layer ...
        & nnet.layer.Formattable
    properties (State)
        log_det
    end
    methods (Abstract)
        inverse(this, inputs, conditions);
    end
end
