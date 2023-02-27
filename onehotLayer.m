classdef onehotLayer < nnet.layer.Layer
    properties
        NumActions
    end
    methods
        function this = onehotLayer(numActions,varargin)
            parser = inputParser;
            addParameter(parser,'Name',"indictator",...
                @(val)validateattributes(val,{'string','char'},{'scalartext'},'','Name'));
            addParameter(parser,'Description',"Convert action integer to indicator vector",...
                @(val)validateattributes(val,{'string','char'},{'scalartext'},'','Description'));
            parse(parser,varargin{:});
            r = parser.Results;
            
            this.Type = "oneHotLayer";
            this.Name = r.Name;
            this.Description = r.Description;
            this.NumInputs = 1;
            this.NumOutputs = 1;
            this.NumActions = numActions;
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = gather(X);
            Z = onehotencode(Z,1,'ClassNames',1:layer.NumActions);
            Z = cast(Z,'like',X);
        end
        
        function dLdX = backward(~,~,~,dLdZ,~)
            % Backward propagate the derivative of the loss function through 
            % the layer
            dLdX = cast(zeros(1,size(dLdZ,2)),'like',dLdZ);
        end
    end
end