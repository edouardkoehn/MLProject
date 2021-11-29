#main file for the submission
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    include("utils.jl")
    test=CSV.read(joinpath(@__DIR__, "..", "data", "testdata.csv"), DataFrame)
    dropmissing!(test)

    train=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    dropmissing!(train)
    coerce!(train,:precipitation_nextday => Multiclass)
    train=train[shuffle(1:size(train, 1)),:]
end

#Benchmark submission KNN K=34, trained on all the data
begin
    model_sub_1=KNNClassifier(K = 34,algorithm = :kdtree,metric = Euclidean(0.0),leafsize = 10,reorder = true,weights = Uniform())
    machine_sub_1=machine(model_sub_1,select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
end
begin
    writeSubmission(test, machine_sub_1,"Sumission_KNN(K=34)_1.0.csv")
end

#Benchmark submission XGB1 (optimized XGB) ,update trained on all the data
begin
    using MLJXGBoostInterface
    model_sub_2=XGBoostClassifier(num_round = 400,booster = "gbtree",disable_default_eval_metric = 0,
        eta = 0.05477225575051663,gamma = 0.0,max_depth = 2,
        min_child_weight = 1.0,max_delta_step = 0.0,subsample = 1.0,
        colsample_bytree = 1.0,colsample_bylevel = 1.0,
        lambda = 1.0,alpha = 0.0,tree_method = "auto",sketch_eps = 0.03,
        scale_pos_weight = 1.0,updater = "auto",refresh_leaf = 1,
        process_type = "default",grow_policy = "depthwise",max_leaves = 0,
        max_bin = 256,predictor = "cpu_predictor",sample_type = "uniform",normalize_type = "tree",
        rate_drop = 0.0,one_drop = 0,skip_drop = 0.0,feature_selector = "cyclic",top_k = 0,
        tweedie_variance_power = 1.5,objective = "automatic",base_score = 0.5, eval_metric = "mlogloss",
        seed = 0,nthread = 1)

    machine_sub_2=machine(model_sub_2,select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> MLJ.fit!
end
begin
    writeSubmission(test, machine_sub_2,"Sumission_XGB_1.1.csv")
end

