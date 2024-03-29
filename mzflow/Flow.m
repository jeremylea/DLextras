classdef Flow
    properties
        data_columns
        input_dim
        conditional_columns
        condition_means
        condition_stds
        autoscale_conditions
        latent
        bijector
    end
    methods
        function this = Flow(data_columns, conditional_columns, bijector, latent, autoscale_conditions)
            if isempty(data_columns)
                error("You must provide data_columns.")
            end
            this.data_columns = data_columns;
            this.input_dim = length(this.data_columns);
            if nargin < 5 || isempty(autoscale_conditions)
                this.autoscale_conditions = true;
            else
                this.autoscale_conditions = ~~autoscale_conditions;
            end
            if nargin < 4 || isempty(latent)
                this.latent = distributions.Uniform(this.input_dim);
            else
                this.latent = latent;
            end
            if this.latent.input_dim ~= length(data_columns)
                error("The latent distribution has %d dimensions, but data_columns has %d dimensions. They must match!",this.latent.input_dim,length(data_columns));
            end
            if nargin >= 3 && ~isempty(bijector)
                set_bijector(this,bijector);
            end
            if nargin < 2 || isempty(conditional_columns)
                this.conditional_columns = [];
                this.condition_means = [];
                this.condition_stds = [];
            else
                this.conditional_columns = conditional_columns;
                this.condition_means = zeros(1,length(this.conditional_columns));
                this.condition_stds = ones(1,length(this.conditional_columns));
            end
        end

        function this = set_bijector(this, bijector)
            this.bijector = bijector;
        end

        function this = set_default_bijector(this, inputs, K, options)
            arguments
                this
                inputs
                K (1,1) {mustBeInteger, mustBeNonnegative, mustBeNonsparse} = 16
                options.hidden_layers (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 1
                options.hidden_dim (1,1) {mustBeInteger, mustBePositive, mustBeNonsparse} = 2*(K+1)*numel(this.data_columns)*max(1,numel(this.conditional_columns))
            end
            data = inputs{:,this.data_columns};
            mins = floor(min(data,[],1));
            maxs = ceil(max(data,[],1));
            has_conditions = numel(this.conditional_columns) > 0;
            layers = layerGraph(featureInputLayer(this.input_dim,name='inputs'));
            if has_conditions
                layers = addLayers(layers,[
                    featureInputLayer(numel(this.conditional_columns),name='conditions')
                ]);
            end
            layers = addLayers(layers,[
                bijectors.ShiftBounds(mins,maxs,name='shift')
            ]);
            layers = connectLayers(layers,'inputs','shift');
            for i = 1:this.input_dim
                sn = ['nsp' num2str(i)];
                rn = ['roll' num2str(i)];
                layers = addLayers(layers,[
                    bijectors.NeuralSplineCoupling(this.input_dim,numel(this.conditional_columns),K,options.hidden_layers,options.hidden_dim,1,name=sn) % match default shift
                    bijectors.Roll(name=rn)
                ]);
                if ~has_conditions
                    if i == 1
                        layers = connectLayers(layers,'shift/out','nsp1');
                    else
                        layers = connectLayers(layers,['roll' num2str(i-1)],sn);
                    end
                else
                    if i == 1
                        layers = connectLayers(layers,'shift/out','nsp1/in');
                    else
                        layers = connectLayers(layers,['roll' num2str(i-1) '/out'],[sn '/in']);
                    end
                    layers = connectLayers(layers,'conditions',[sn '/cond']);
                end
            end
            this = set_bijector(this,dlnetwork(layers));
        end

        function check_bijector(this)
            if isempty(this.bijector)
                error("The bijector has not been set up yet! You can do this by calling flow.set_bijector(bijector), or by calling train, in which case the default bijector will be used.");
            end
        end

        function conditions = get_conditions(this, inputs)
            if isempty(this.conditional_columns)
                conditions = zeros(size(inputs,1),0);
            else
                conditions = inputs{:,this.conditional_columns};
                conditions = (conditions - this.condition_means) ./ this.condition_stds;
            end
        end

        function [neg_log_prob,log_probs] = neg_log_prob(this, inputs)
            check_bijector(this);
            X = dlarray(inputs{:,this.data_columns},"BC");
            conditions = dlarray(get_conditions(this,inputs),"BC");
            [~,log_probs] = log_probs_internal(this.bijector,this.latent,X,conditions);
            neg_log_prob = -mean(log_probs,"all");
        end

        function pdfs = posterior(this, inputs, column, grid, normalize, batch_size)
            check_bijector(this)
            idx = find(strcmp(this.data_columns,column));
            this.data_columns(idx) = []; % we're not returning this
            if isempty(this.data_columns) && isempty(this.conditional_columns)
                nrows = 1;
            else
                nrows = size(inputs,1);
            end
            grid = grid(:);
            if nargin < 6 || isempty(batch_size)
                batch_size = nrows;
            end
            pdfs = zeros(nrows,numel(grid));
            for i = 1:batch_size:nrows
                batch_idx=i:min(i+batch_size-1,nrows);
                batch = inputs(batch_idx,:);
                conditions = get_conditions(this,batch);
                batch = batch{:,this.data_columns};
                batch = [repmat(batch(:,1:idx-1),numel(grid),1);...
                    repelem(grid,size(batch,1),1);...
                    repmat(batch(:,idx:end),numel(grid),1)];
                conditions = repmat(conditions,numel(grid),1);
                batch = dlarray(batch,"BC");
                conditions = dlarray(conditions,"BC");
                [~,log_probs] = log_probs_internal(this.bijector,this.latent,batch,conditions);
                log_probs = reshape(log_probs,[],numel(grid));
                pdfs(batch_idx,:) = exp(log_probs);
            end
            if nargin < 5 || isempty(normalize) || normalize
                pdfs = pdfs./sum(pdfs,2);
            end
            pdfs(isnan(pdfs)) = 0;
        end

        function [x, u] = sample(this, nsamples, conditions, save_conditions, u)
            check_bijector(this);
            if nargin < 5
                u = [];
            elseif ~isempty(u)
                nsamples = size(u,1)/max(1,size(conditions,1));
            end
            if nargin < 4 || isempty(save_conditions)
                save_conditions = true;
            end
            if ~isempty(this.conditional_columns) && isempty(conditions)
                error("Must provide the following conditions\n%s",this.conditional_columns);
            end
            if isempty(this.conditional_columns)
                conditions = zeros(nsamples,0);
            else
                conditions = get_conditions(this,conditions);
                conditions = repelem(conditions,nsamples,1);
            end
            if isempty(u)
                u = this.latent.sample(size(conditions,1));
            end
            x = dlarray(u,"BC");
            conditions = dlarray(conditions,"BC");
            for i = numel(this.bijector.Layers):-1:(2+(numel(this.conditional_columns)>0))
                l = this.bijector.Layers(i);
                if ~isa(l,"bijectors.Bijector")
                    continue;
                end
                x = this.bijector.Layers(i).inverse(x,conditions);
            end
            x = extractdata(x)';
            conditions = extractdata(conditions)';
            if isempty(this.conditional_columns) || ~save_conditions
                x = array2table(x,"VariableNames",this.data_columns);
            else
                conditions = conditions .* this.condition_stds + this.condition_means;
                x = array2table([x conditions],"VariableNames",[this.data_columns this.conditional_columns]);
            end
        end

        function [this, losses, val_losses] = train(this, inputs, validation, epochs, batch_size, debug_cb, varargin)
            if nargin < 6, debug_cb = []; end
            if nargin < 5 || isempty(batch_size) || batch_size < 1, batch_size = 1000; end
            if nargin < 4 || isempty(epochs) || epochs < 1, epochs = 100; end
            if nargin < 3, validation = []; end
            if isempty(this.bijector)
                this = set_default_bijector(this,inputs);
            end
            if ~isempty(this.conditional_columns) && this.autoscale_conditions
                C = get_conditions(this,inputs);
                this.condition_means = mean(C);
                this.condition_stds = arrayfun(@(x) x ~= 0,std(C));
            end
            ds = arrayDatastore(inputs(:,[this.data_columns this.conditional_columns]),IterationDimension=1);
            mbq = minibatchqueue(ds,2,...
                MiniBatchSize=batch_size,...
                MiniBatchFcn=@(X) preprocessData(this,X),...
                MiniBatchFormat=["BC" "BC"]);
            losses = [];
            val_losses = [];
            iteration = 0;
            trailingAvg = [];
            trailingAvgSq = [];
            initialLearnRate = 0.1*min(1,batch_size/1000);
            decay = 0.2;
            for epoch = 0:epochs
                if epoch > 0
                    learnRate = initialLearnRate*decay.^log10(epoch);
                    shuffle(mbq);
                    while hasdata(mbq)
                        iteration = iteration+1;
                        [Xbat,Cbat] = next(mbq);
                        [~,gradients] = dlfeval(@loss_fun,this.bijector,this.latent,Xbat,Cbat);
                        [this.bijector,trailingAvg,trailingAvgSq] = adamupdate(this.bijector,gradients,...
                            trailingAvg,trailingAvgSq,iteration,learnRate);
                    end
                end
                if ~isempty(debug_cb)
                    debug_cb(this,varargin{:});
                end
                losses(end+1) = neg_log_prob(this,inputs); %#ok<AGROW>
                if ~isempty(validation)
                    val_losses(end+1) = neg_log_prob(this,validation); %#ok<AGROW>
                    fprintf("Epoch: %i, Loss: %f\n, Validation Loss: %f",epoch,losses(end),val_losses(end));
                else
                    fprintf("Epoch: %i, Loss: %f\n",epoch,losses(end));
                end
                if ~isfinite(losses(end))
                    disp(["Training stopping after epoch ",num2str(epoch)," because training loss diverged."]);
                    break;
                end
            end
        end
    end
    methods (Access = private)
        function [X,C] = preprocessData(this, batch)
            batch = cat(1,batch{:});
            X = batch{:,this.data_columns};
            C = get_conditions(this,batch);
        end
    end
end

function [loss,gradients] = loss_fun(bijector, latent, x, c)
    [bijector,log_probs] = log_probs_internal(bijector,latent,x,c);
    loss = -mean(log_probs,"all");
    gradients = dlgradient(loss,bijector.Learnables);
end

function [bijector, log_probs] = log_probs_internal(bijector, latent, x, conditions)
    if isempty(conditions)
        [x,state] = predict(bijector,x);
    else
        [x,state] = predict(bijector,x,conditions);
    end
    bijector.State = state;
    log_probs = latent.log_prob(x);
    for i = 1:height(state)
        if strcmp(state{i,"Parameter"},"log_det")
            log_probs = log_probs+state{i,"Value"}{1};
        end
    end
    log_probs(isnan(log_probs)) = -inf;
end
