rng("default") % For reproducibility
[X,label] = twomoons(200);
data = array2table([X label],"VariableNames",{"x","y","label"});
test = array2table(unique(label),"VariableNames",{"label"});

%scatter(X(:,1),X(:,2),[],label,"filled")
%title("Labeled Data")
flow = Flow(["x", "y"],"label");
flow = flow.set_default_bijector(data);
flow.sample(1,test);
[flow,losses] = flow.train(data);
samples = flow.sample(10,test);
%grid = jnp.linspace(-2, 2, 100);
%pdfs = flow.posterior(data, column="x", grid=grid);