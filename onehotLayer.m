classdef onehotLayer < nnet.layer.Layer ...
        & nnet.layer.Acceleratable ...
        & nnet.layer.Formattable
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
            d = finddim(X,"C");
            sz = size(X);
            sz(d) = this.NumEncoded;
            Z = dlarray(zeros(sz),dims(X));
            if d ~= 1
                p = 1:numel(sz);
                p(p == d) = [];
                p = [d p];
                t = dims(X);
                t = ["C" strrep(t,"C","")];
                X = permute(stripdims(X),p);
                Z = permute(stripdims(Z),p);
            end
            S.subs = repmat({':'},1,ndims(X));
            S.type = '()';
            R = S;
            for j = 1:length(this.NumActions)
                if this.NumActions(j) == 1
                    R.subs{1} = j;
                    S.subs{1} = i+1;
                    Z = subsasgn(Z,S,subsref(X,R));
                    i = i+1;
                else
                    R.subs{1} = j;
                    S.subs{1} = i+(1:this.NumActions(j));
                    Z = subsasgn(Z,S,onehotencode(subsref(X,R),1,'ClassNames',1:this.NumActions(j)));
                    i = i+this.NumActions(j);
                end
            end
            if d ~= 1
                Z = dllarray(Z,t);
            end
            Z = cast(Z,'like',X);
        end
    end
end