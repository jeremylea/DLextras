classdef onehotLayer < nnet.layer.Layer
    properties
        NumActions
        NumEncoded
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
            i = 0;
            for j = 1:length(this.NumActions)
                if this.NumActions(j) == 1
                    i = i+1;
                else
                    i = i+this.NumActions(j);
                end
            end
            this.NumEncoded = i;
        end
        
        function Z = predict(this, X)
            i = 0;
            Z = zeros(this.NumEncoded,size(X,2));
            for j = 1:length(this.NumActions)
                if this.NumActions(j) == 1
                    Z(i+1,:) = X(j,:);
                    i = i+1;
                else
                    Z(i+1:this.NumActions(j),:) = onehotencode(X(j,:),1,'ClassNames',1:this.NumActions(j));
                    i = i+this.NumActions(j);
                end
            end
            Z = cast(Z,'like',X);
        end
    end
end