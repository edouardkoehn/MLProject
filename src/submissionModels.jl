### Main file for the submission
# In this files, we generate the submission for Kaggle. Each model is explicitely defined and trained on the complete dataset.

### Import the data
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    include("utils.jl")
end

begin
    #import test
    test=importTest()
    #import cleaned train
    train=importTrainCleaned()
    #import filled train
    train_filled=importTrainFilled()
    #import augmented data
    augmented=importAugmented()
end

## Benchmark submission, LogisticClassifier L2, trained on cleaned data
begin
    model_sub_3=LogisticClassifier(lambda = 774263.6826811269,
            gamma = 0.0,
            penalty = :l2,
            fit_intercept = true,
            penalize_intercept = false,
            solver = nothing)
    machine_sub_3=machine(model_sub_3,select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
end
begin
    writeSubmission(test, machine_sub_3,"Sumission_LR_L2(cleaned).csv")
end
begin
    model_sub_11=LogisticClassifier(
                lambda = 774263.6826811269,
                gamma = 0.0,
                penalty = :l2,
                fit_intercept = true,
                penalize_intercept = false,
                solver = nothing)
    machine_sub_11=machine(model_sub_11,select(train_filled, Not(:precipitation_nextday)),train_filled.precipitation_nextday)|> fit!
end
begin
    writeSubmission(test, machine_sub_11,"Sumission_LR_L2(filled).csv")
end

## Benchmark submission KNN K=34, trained on the cleaned data
begin
    model_sub_1=KNNClassifier(K = 34,algorithm = :kdtree,
                                metric = Euclidean(0.0),
                                leafsize = 10,
                                reorder = true,
                                weights = Uniform())
    machine_sub_1=machine(model_sub_1,select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
end
begin
    writeSubmission(test, machine_sub_1,"Sumission_KNN(K=34)_1.0.csv")
end

begin
    model_sub_12=KNNClassifier(
                            K = 25,
                            algorithm = :kdtree,
                            metric = Euclidean(0.0),
                            leafsize = 10,
                            reorder = true,
                            weights = Uniform())
    machine_sub_12=machine(model_sub_12,select(train_filled, Not(:precipitation_nextday)),train_filled.precipitation_nextday)|> fit!
end
begin
    writeSubmission(test, machine_sub_12,"Sumission_KNN(K=25)_filled.csv")
end



## Benchmark submission XGB1 (optimized XGB) , trained on all the data
begin
    model_sub_2=XGBoostClassifier(num_round = 400,booster = "gbtree",
                                disable_default_eval_metric = 0,
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

## Benchmark sumission NN optimised,trained on all the data
begin
    model_sub_4=NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 125,dropout = 0.2,σ = relu),
                                        optimiser = ADAM(0.001, (0.9, 0.999), IdDict{Any,Any}()),
                                        loss = Flux.crossentropy,
                                        batch_size = 500,
                                        lambda = 0.0,
                                        alpha = 0.0,
                                        epochs= 1000,
                                        optimiser_changes_trigger_retraining = false)
    machine_sub_4= machine(model_sub_4,select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> MLJ.fit!
end
begin
    writeSubmission(test, machine_sub_4, "Submission_NN_opt.csv")
end

## Benchmark sumission RandomForest n=383,trained on all the data
begin
    model_sub_5=RandomForestClassifier(
        max_depth = -1,
        min_samples_leaf = 1,
        min_samples_split = 2,
        min_purity_increase = 0.0,
        n_subfeatures = -1,
        n_trees = 383,
        sampling_fraction = 0.7,
        pdf_smoothing = 0.0)
    machine_sub_5= machine(model_sub_5,select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> MLJ.fit!
end
begin
    writeSubmission(test, machine_sub_5, "Submission_RandomForest_383.csv")
end

## Benchmark sumission XGBBoost1 with filled data,trained on all the data
begin
    model_sub_6=XGBoostClassifier(num_round = 500,
                                    booster = "gbtree",
                                    disable_default_eval_metric = 0,
                                    eta = 0.10000000000000002,
                                    gamma = 0.0,
                                    max_depth = 4,
                                    min_child_weight = 1.0,
                                    max_delta_step = 0.0,
                                    subsample = 1.0,
                                    colsample_bytree = 1.0,
                                    colsample_bylevel = 1.0,
                                    lambda = 1.0,
                                    alpha = 0.0,
                                    tree_method = "auto",
                                    sketch_eps = 0.03,
                                    scale_pos_weight = 1.0,
                                    updater = "auto",
                                    refresh_leaf = 1,
                                    process_type = "default",
                                    grow_policy = "depthwise",
                                    max_leaves = 0,
                                    max_bin = 256,
                                    predictor = "cpu_predictor",
                                    sample_type = "uniform",
                                    normalize_type = "tree",
                                    rate_drop = 0.0,
                                    one_drop = 0,
                                    skip_drop = 0.0,
                                    feature_selector = "cyclic",
                                    top_k = 0,
                                    tweedie_variance_power = 1.5,
                                    objective = "automatic",
                                    base_score = 0.5,
                                    eval_metric = "mlogloss",
                                    seed = 0,
                                    nthread = 1)

    machine_sub_6=machine(model_sub_6,select(train_filled, Not(:precipitation_nextday)), train_filled.precipitation_nextday)|>MLJ.fit!
end
begin
    writeSubmission(test, machine_sub_6, "Submission_XGB_filled_1.0.csv")
end
## Benchmark sumission XGBBoost1 with filled data and std,trained on all the data
begin
    model_sub_7=XGBoostClassifier(
                                num_round = 600,
                                booster = "gbtree",
                                disable_default_eval_metric = 0,
                                eta = 0.010000000000000004,
                                gamma = 0.0,
                                max_depth = 10,
                                min_child_weight = 1.0,
                                max_delta_step = 0.0,
                                subsample = 0.5,
                                colsample_bytree = 0.5,
                                colsample_bylevel = 1.0,
                                lambda = 1.0,
                                alpha = 0.0,
                                tree_method = "auto",
                                sketch_eps = 0.03,
                                scale_pos_weight = 1.0,
                                updater = "auto",
                                refresh_leaf = 1,
                                process_type = "default",
                                grow_policy = "depthwise",
                                max_leaves = 0,
                                max_bin = 256,
                                predictor = "cpu_predictor",
                                sample_type = "uniform",
                                normalize_type = "tree",
                                rate_drop = 0.0,
                                one_drop = 0,
                                skip_drop = 0.0,
                                feature_selector = "cyclic",
                                top_k = 0,
                                tweedie_variance_power = 1.5,
                                objective = "automatic",
                                base_score = 0.5,
                                eval_metric = "mlogloss",
                                seed = 0,
                                nthread = 1)

        train_filled_std = MLJ.transform(fit!(machine(Standardizer(),select(train_filled, Not(:precipitation_nextday)))))
        machine_sub_7=machine(model_sub_7,train_filled_std, train_filled.precipitation_nextday)|>MLJ.fit!
end
begin
    test_std= MLJ.transform(fit!(machine(Standardizer(),test)))
    writeSubmission(test_std, machine_sub_7, "Submission_XGB_filled_std_1.0.csv")
end
## Benchmark submission XGBBoost1 with augmented_data, trained on all the data
begin
    model_sub_8=XGBoostClassifier(
                                num_round = 500,
                                booster = "gbtree",
                                disable_default_eval_metric = 0,
                                eta = 0.10000000000000002,
                                gamma = 0.0,
                                max_depth = 6,
                                min_child_weight = 1.0,
                                max_delta_step = 0.0,
                                subsample = 1.0,
                                colsample_bytree = 1.0,
                                colsample_bylevel = 1.0,
                                lambda = 1.0,
                                alpha = 0.0,
                                tree_method = "auto",
                                sketch_eps = 0.03,
                                scale_pos_weight = 1.0,
                                updater = "auto",
                                refresh_leaf = 1,
                                process_type = "default",
                                grow_policy = "depthwise",
                                max_leaves = 0,
                                max_bin = 256,
                                predictor = "cpu_predictor",
                                sample_type = "uniform",
                                normalize_type = "tree",
                                rate_drop = 0.0,
                                one_drop = 0,
                                skip_drop = 0.0,
                                feature_selector = "cyclic",
                                top_k = 0,
                                tweedie_variance_power = 1.5,
                                objective = "automatic",
                                base_score = 0.5,
                                eval_metric = "mlogloss",
                                seed = 0,
                                nthread = 1)
        machine_sub_8=machine(model_sub_8,select(augmented, Not(:precipitation_nextday)), augmented.precipitation_nextday)|>MLJ.fit!
end
begin
    writeSubmission(test, machine_sub_8, "Submission_XGB_augmented_1.0.csv")
end