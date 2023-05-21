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
                error('You must provide data_columns.')
            end
            this.data_columns = data_columns;
            this.input_dim = length(this.data_columns);
            if nargin < 5 || isempty(autoscale_conditions)
                this.autoscale_conditions = false;
            else
                this.autoscale_conditions = ~~autoscale_conditions;
            end
            if nargin < 4 || isempty(latent)
                this.latent = distributions.Uniform(this.input_dim);
            else
                this.latent = latent;
            end
            if this.latent.input_dim ~= length(data_columns)
                error('The latent distribution has %d dimensions, but data_columns has %d dimensions. They must match!', this.latent.input_dim, length(data_columns));
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
                this.condition_means = zeros(1, length(this.conditional_columns));
                this.condition_stds = ones(1, length(this.conditional_columns));
            end
        end

        function this = set_bijector(this, bijector)
            this.bijector = bijector;
        end

        function this = set_default_bijector(this, inputs)
            data = inputs{:, this.data_columns};
            mins = floor(min(data, [], 1));
            maxs = ceil(max(data, [], 1));
            condition_dim = numel(this.conditional_columns) > 0;
            layers = layerGraph(featureInputLayer(this.input_dim,name='inputs'));
            if condition_dim
                layers = addLayers(layers,[
                    featureInputLayer(numel(this.conditional_columns),name='conditions')
                ]);
            end
            layers = addLayers(layers,[
                bijectors.ShiftBounds(mins, maxs, name='shift')
            ]);
            layers = connectLayers(layers,'inputs','shift');
            for i = 1:this.input_dim
                sn = ['nsp' num2str(i)];
                an = ['accn' num2str(i)];
                rn = ['roll' num2str(i)];
                bn = ['accr' num2str(i)];
                layers = addLayers(layers,[
                    bijectors.NeuralSplineCoupling(this.input_dim,condition_dim,name=sn)
                    bijectors.Roll(name=rn)
                ]);
                layers = addLayers(layers,[
                    bijectors.Accumulate(name=an)
                    bijectors.Accumulate(name=bn)
                ]);
                layers = connectLayers(layers,[sn '/log_det'],[an '/in2']);
                layers = connectLayers(layers,[rn '/log_det'],[bn '/in2']);
                if ~condition_dim
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
                if i == 1
                    layers = connectLayers(layers,'shift/log_det',[an '/in1']);
                else
                    layers = connectLayers(layers,['accr' num2str(i-1)],[an '/in1']);
                end
            end
            this = set_bijector(this,dlnetwork(layers));
        end

        function check_bijector(this)
            if isempty(this.bijector)
                error('The bijector has not been set up yet! You can do this by calling flow.set_bijector(bijector), or by calling train, in which case the default bijector will be used.');
            end
        end

        function conditions = get_conditions(this, inputs)
            if isempty(this.conditional_columns)
                conditions = zeros(size(inputs, 1),0);
            else
                conditions = inputs{:, this.conditional_columns};
                conditions = (conditions - this.condition_means) ./ this.condition_stds;
            end
        end

        function log_probs = neg_log_prob(this, inputs)
            check_bijector(this);
            X = dlarray(inputs{:, this.data_columns},"BC");
            conditions = dlarray(get_conditions(this,inputs),"BC");
            [~, log_probs] = log_probs_internal(this.bijector,this.latent,X,conditions);
            log_probs = -mean(log_probs,"all");
        end

        function pdfs = posterior(this, inputs, column, grid, marg_rules, normalize, batch_size)
            check_bijector(this)
            idx = find(strcmp(this.data_columns, column));
            this.data_columns(idx) = [];
            nrows = size(inputs, 1);
            if nargin < 7 || isempty(batch_size)
                batch_size = size(inputs, 1);
            end
            inputs = reset_index(inputs, 'drop');
            pdfs = zeros(nrows, numel(grid));
            if ~isempty(marg_rules)
                if isnan(marg_rules.flag)
                    check_flags = @(data) isnan(data);
                else
                    check_flags = @(data) isclose(data, marg_rules.flag);
                end
                unflagged_idx = find(~any(check_flags(inputs(:, this.data_columns)), 2));
                unflagged_pdfs = posterior(this, inputs(unflagged_idx, :), column, grid, [], false, batch_size);
                pdfs(unflagged_idx, :) = unflagged_pdfs;
                already_done = unflagged_idx;

                % Iterate over marginalization rules
                for name = fieldnames(marg_rules)'
                    rule = marg_rules.(name);
                    if strcmp(name,'flag')
                        continue;
                    end
                    % Identify flagged rows and apply marginalization
                    flagged_idx = find(any(check_flags(inputs.(name)), 2));
                    flagged_idx = setdiff(flagged_idx, already_done);
                    if isempty(flagged_idx)
                        continue;
                    end
                    marg_grids = arrayfun(rule, inputs(flagged_idx, :), 'UniformOutput', false);
                    marg_grids = cat(2, marg_grids{:});
                    marg_inputs = repmat(inputs(flagged_idx, :), size(marg_grids, 2), 1);
                    marg_inputs.(name) = marg_grids(:);
                    marg_pdfs = posterior(this, marg_inputs, column, grid, marg_rules, false, batch_size);
                    marg_pdfs = reshape(marg_pdfs, [length(flagged_idx), size(marg_grids, 2), length(grid)]);
                    marg_pdfs = sum(marg_pdfs, 2);
                    pdfs(flagged_idx, :) = marg_pdfs;
                    already_done(end+1) = flagged_idx; %#ok<AGROW>
                end
            else
                for batch_idx = 1:batch_size:nrows
                    batch = inputs(batch_idx:min(batch_idx+batch_size-1, nrows), :);
                    conditions = get_conditions(this,batch);
                    batch = batch{:, this.data_columns};
                    batch = [repmat(batch(:, 1:idx-1), 1, numel(grid));...
                        repelem(grid, size(batch, 1), 1);...
                        repmat(batch(:, idx+1:end), 1, numel(grid))];
                    conditions = repmat(conditions, numel(grid), 1);
                    log_prob = log_probs_internal(this,batch, conditions);
                    log_prob = reshape(log_prob, [], numel(grid));
                    prob = exp(log_prob);
                    pdfs(batch_idx:min(batch_idx+batch_size-1,nrows), :) = prob;
                end
            end
            if nargin < 6 || isempty(normalize) || normalize
                pdfs = pdfs./trapz(grid, pdfs, 2);
            end
            pdfs(isnan(pdfs)) = 0;
        end

        function x = sample(this, nsamples, conditions, save_conditions)
            check_bijector(this);
            if nargin < 4 || isempty(save_conditions)
                save_conditions = true;
            end
            if ~isempty(this.conditional_columns) && isempty(conditions)
                error("Must provide the following conditions\n%s", this.conditional_columns);
            end
            if isempty(this.conditional_columns)
                conditions = zeros(nsamples, 0);
            else
                conditions = get_conditions(this,conditions);
                conditions = repelem(conditions, nsamples, 1);
            end
            x = this.latent.sample(size(conditions,1));
            x = dlarray(x,"BC");
            conditions = dlarray(conditions,"BC");
            for i = numel(this.bijector.Layers):-1:(2+(numel(this.conditional_columns)>0))
                l = this.bijector.Layers(i);
                if ~isa(l,"bijectors.Bijector")
                    continue;
                end
                x = this.bijector.Layers(i).inverse(x, conditions);
            end
            x = extractdata(x)';
            conditions = extractdata(conditions)';
            if isempty(this.conditional_columns) || ~save_conditions
                x = array2table(x, 'VariableNames', this.data_columns);
            else
                conditions = conditions .* this.condition_stds + this.condition_means;
                x = array2table([x conditions], 'VariableNames', [this.data_columns this.conditional_columns]);
            end
        end

        function [this, losses, val_losses] = train(this, inputs, validation, epochs, batch_size)
            if nargin < 5, batch_size = 1024; end
            if nargin < 4, epochs = 100; end
            if nargin < 3, validation = []; end
            if isempty(this.bijector)
                this = set_default_bijector(this,inputs);
            end
            if ~isempty(this.conditional_columns) && this.autoscale_conditions
                C = get_conditions(this,inputs);
                this.condition_means = mean(C);
                this.condition_stds = arrayfun(@(x) x ~= 0, std(C));
            end
            ds = arrayDatastore(inputs(:,[this.data_columns this.conditional_columns]),IterationDimension=1);
            mbq = minibatchqueue(ds,2,...
                MiniBatchSize=batch_size,...
                MiniBatchFcn=@(X) preprocessData(this,X),...
                MiniBatchFormat=["BC" "BC"]);
            losses = neg_log_prob(this,inputs);
            if ~isempty(validation)
                val_losses = neg_log_prob(this,validation);
            else
                val_losses = [];
            end
            iteration = 0;
            trailingAvg = [];
            trailingAvgSq = [];
            for epoch = 1:epochs
                shuffle(mbq);
                while hasdata(mbq)
                    iteration = iteration+1;
                    [Xbat,Cbat] = next(mbq);
                    [~,gradients] = dlfeval(@loss_fun,this.bijector,this.latent,Xbat,Cbat);
                    [this.bijector,trailingAvg,trailingAvgSq] = adamupdate(this.bijector,gradients,...
                        trailingAvg,trailingAvgSq,iteration);
                end
                losses(end+1) = neg_log_prob(this,inputs); %#ok<AGROW>
                if ~isempty(validation)
                    val_losses(end+1) = neg_log_prob(this,validation); %#ok<AGROW>
                end
                if ~isfinite(losses(end))
                    disp(['Training stopping after epoch ', num2str(epoch+1), ' because training loss diverged.']);
                    break;
                end
            end
        end
    end
    methods (Access = private)
        function [X,C] = preprocessData(this, batch)
            batch = cat(1,batch{:});
            X = batch{:, this.data_columns};
            C = get_conditions(this,batch);
        end
    end
end

function [loss,gradients] = loss_fun(bijector, latent, x, c)
    [bijector, log_probs] = log_probs_internal(bijector,latent,x,c);
    loss = -mean(log_probs,"all");
    gradients = dlgradient(loss,bijector.Learnables);
end

function [bijector, log_probs] = log_probs_internal(bijector, latent, x, conditions)
    [x,log_det] = predict(bijector,x,conditions);
    log_probs = latent.log_prob(x)+log_det;
    log_probs(isnan(log_probs)) = -inf;
end