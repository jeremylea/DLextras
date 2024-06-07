classdef Roll < bijectors.Bijector
    properties
        shift
    end
    methods
        function this = Roll(shift,options)
            arguments
                shift (1,1) {mustBeInteger, mustBeNonsparse} = 1
                options.Name {mustBeText} = ''
                options.Description {mustBeText} = strjoin(["Roll (circular shift)" num2str(shift)])
            end
            this.Type = "Roll";
            this.Name = char(options.Name);
            this.Description = options.Description;
            this.shift = shift;
        end
        function [outputs, log_det, penalty] = predict(this, inputs)
            outputs = circshift(inputs,this.shift,finddim(inputs,"C"));
            log_det = zeros(1,size(inputs,finddim(inputs,"B")));
            log_det = stripdims(dlarray(log_det));
            penalty = zeros(1,size(inputs,finddim(inputs,"B")));
            penalty = stripdims(dlarray(penalty));
        end
        function outputs = inverse(this, inputs, ~)
            outputs = circshift(inputs,-this.shift,finddim(inputs,"C"));
        end
    end
end
