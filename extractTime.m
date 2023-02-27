classdef extractTime < nnet.layer.Layer ...
        & nnet.layer.Acceleratable ...
        & nnet.layer.Formattable
    properties
        Time
    end
    methods
        function this = extractTime(Time,varargin)
            parser = inputParser;
            addParameter(parser,'Name',"extract",...
                @(val)validateattributes(val,{'string','char'},{'scalartext'},'','Name'));
            addParameter(parser,'Description',"Extract one time from a sequence",...
                @(val)validateattributes(val,{'string','char'},{'scalartext'},'','Description'));
            parse(parser,varargin{:});
            r = parser.Results;
            
            this.Type = "extractTime";
            this.Name = r.Name;
            this.Description = r.Description;
            this.NumInputs = 1;
            this.NumOutputs = 1;
            if ~isnumeric(Time)
                if ~strcmp(Time,'last')
                    error("Invalid time string");
                end
                Time = string(Time);
            elseif ~isscalar(Time) || ~isinteger(Time)
                error("Time must be an integer scalar");
            elseif Time < 1
                error("Time must be greater than zero");
            end
            this.Time = Time;
        end
        function Z = predict(layer, X)
            d = finddim(X,"T");
            D = dims(X);
            if isempty(d)
                error("Invalid input, does not contain time dim");
            end
            t = size(X,d);
            if ~isstring(layer.Time) && layer.Time > t
                error("Time greater than sequence length")
            end
            C = repmat({':'},1,ndims(X));
            C{d} = t;
            D(d) = [];
            Z = dlarray(X(C{:}),D);
        end
    end
end