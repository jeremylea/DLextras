classdef Accumulate < nnet.layer.Layer ...
        & nnet.layer.Acceleratable ...
        & nnet.layer.Formattable
    methods
        function this = Accumulate(options)
            arguments
                options.Name {mustBeText} = ''
                options.Description {mustBeText} = "Log Determinant accumulator layer"
            end
            this.Type = "Accumulate";
            this.Name = char(options.Name);
            this.Description = options.Description;
            this.NumInputs = 2;
            this.NumOutputs = 1;
        end
        function [log_dets] = predict(~, log_dets, log_det)
            log_dets = log_dets+log_det;
        end
    end
end
