classdef ShiftBounds < bijectors.Bijector
    properties
        min_val
        max_val
    end
    properties (Access = private)
        range
    end
    methods
        function this = ShiftBounds(min_val, max_val, options)
            arguments
                min_val {mustBeReal, mustBeNonsparse}
                max_val {mustBeReal, mustBeNonsparse}
                options.Name {mustBeText} = ''
                options.Description {mustBeText} = strjoin(["ShiftBounds" num2str(min_val)  num2str(max_val)])
            end
            this.Type = "ShiftBounds";
            this.Name = char(options.Name);
            this.Description = options.Description;
            this.min_val = dlarray(min_val(:),"C");
            this.max_val = dlarray(max_val(:),"C");
            this.range = dlarray(max_val(:) - min_val(:),"C");
        end
        function [outputs, log_det, penalty] = predict(this, inputs)
            outputs = (inputs - this.min_val) ./ this.range;
            log_det = log(prod(1./this.range))...
                .*ones(1,size(inputs,finddim(inputs,"B")));
            log_det = stripdims(dlarray(log_det));
            penalty = zeros(1,size(inputs,finddim(inputs,"B")));
            penalty = stripdims(dlarray(penalty));
        end
        function outputs = inverse(this, inputs, ~)
            outputs = inputs .* this.range + this.min_val;
        end
    end
end
