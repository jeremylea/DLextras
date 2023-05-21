classdef Roll < bijectors.Bijector
    properties
        shift
    end
    methods
        function this = Roll(shift,options)
            arguments
                shift (1,1) {mustBeInteger, mustBeNonsparse} = 1
                options.Name {mustBeText} = ''
                options.Description {mustBeText} = "Roll (circular shift) layer"
            end
            this.Type = "Roll";
            this.Name = char(options.Name);
            this.Description = options.Description;
            this.shift = shift;
        end
        function [outputs, log_det] = predict(this, inputs)
            outputs = circshift(inputs,this.shift,finddim(inputs,"C"));
            log_det = zeros(1,size(inputs,finddim(inputs,"B")));
            log_det = stripdims(dlarray(log_det));
        end
        function outputs = inverse(this, inputs, ~)
            outputs = circshift(inputs,-this.shift,finddim(inputs,"C"));
        end
        function [inf] = info(this)
            inf = {"Roll" this.shift};
        end
    end
end
