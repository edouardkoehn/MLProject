### Main file for testing the different linear MLJLinearModels
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    include("utils.jl")
end

### Import of the data test
begin
    #import test
    test=importTest()
    #import cleaned train
    train=importTrainCleaned()
    #import filled train
    train_filled=importTrainFilled()
    #subset for PUY weather station  
    train_PUY= hcat(select(train, r"PUY"), DataFrame(precipitation_nextday=train.precipitation_nextday))
end

### LogisticClassifier 

##  LogisticClassifier (MLJ) ALL No Regularisation
begin
    mach_Log_Classifier_MLJ_NoR= machine(LogisticClassifier(penalty=:none), select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
    evaluate!(mach_Log_Classifier_MLJ_NoR, measure=auc)
end
begin
    exportMachine("mach_Log_Classifier_MLJ_NoR.jlso", mach_Log_Classifier_MLJ_NoR) 
end

## LogisticClassifier (MLJ) ALL L1
begin
    Log_Classifier_MLJ_L1= LogisticClassifier(penalty=:l1)
    self_Log_Classifier_MLJ_L1=TunedModel(model = Log_Classifier_MLJ_L1,
        resampling = CV(nfolds = 10),
        tuning = Grid(),
        range=range(Log_Classifier_MLJ_L1, :lambda,
        lower = 1e-3, upper = 1e7,
        scale = :log),
        measure=auc)
    mach_Log_Classifier_MLJ_L1=machine(self_Log_Classifier_MLJ_L1, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_MLJ_L1)
    evaluate!(mach_Log_Classifier_MLJ_L1, measure=auc)
end
begin
    exportMachine("mach_Log_Classifier_MLJ_L1.jlso", mach_Log_Classifier_MLJ_L1) 
end

## LogisticClassifier (MLJ) ALL L2
begin
    Log_Classifier_MLJ_L2= LogisticClassifier(penalty=:l2)
    self_Log_Classifier_MLJ_L2=TunedModel(model = Log_Classifier_MLJ_L2,
        resampling = CV(nfolds = 10),
        tuning = Grid(),
        range=range(Log_Classifier_MLJ_L2, :lambda,
        lower = 1e-3, upper = 1e7,
        scale = :log),
        measure=auc)
    mach_Log_Classifier_MLJ_L2=machine(self_Log_Classifier_MLJ_L2, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_MLJ_L2)
    evaluate!(mach_Log_Classifier_MLJ_L2, measure=auc)
end
begin
    exportMachine("mach_Log_Classifier_MLJ_L2.jlso", mach_Log_Classifier_MLJ_L2) 
end

## LogisticClassifier (MLJ) PUY No Regularisation
begin
    mach_Log_Classifier_MLJ_PUY=machine(LogisticClassifier(penalty=:none), select(train_PUY, Not(:precipitation_nextday)),train_PUY.precipitation_nextday)|> fit!
    evaluate!(mach_Log_Classifier_PUY, measure=auc)
end
begin
    exportMachine("mach_Log_Classifier_MLJ_PUY_NoR.jlso", mach_Log_Classifier_MLJ_PUY)
end

##LogisticClassifier (MLJ) PUY L1
begin
    Log_Classifier_MLJ_PUY_L1= LogisticClassifier(penalty=:l1)
    self_Log_Classifier_MLJ_PUY_L1=TunedModel(model = Log_Classifier_MLJ_PUY_L1,
        resampling = CV(nfolds = 10),
        tuning = Grid(),
        range=range(Log_Classifier_MLJ_PUY_L1, :lambda,
        lower = 1e-4, upper = 1e2,
        scale = :log),
        measure=auc)

    mach_Log_Classifier_MLJ_PUY_L1=machine(self_Log_Classifier_MLJ_PUY_L1, select(train_PUY, Not(:precipitation_nextday)),train_PUY.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_MLJ_PUY_L1)
    evaluate!(mach_Log_Classifier_PUY_L1, measure=auc)
end
begin
    exportMachine("mach_Log_Classifier_MLJ_PUY_L1.jlso", mach_Log_Classifier_MLJ_PUY_L1)
end

##LogisticClassifier (MLJ) PUY L2
begin
    Log_Classifier_MLJ_PUY_L2= LogisticClassifier(penalty=:l2)
    self_Log_Classifier_MLJ_PUY_L2=TunedModel(model = Log_Classifier_MLJ_PUY_L2,
        resampling = CV(nfolds = 10),
        tuning = Grid(),
        range=range(Log_Classifier_MLJ_PUY_L2, :lambda,
        lower = 1e-2, upper = 1e7,
        scale = :log),
        measure=auc)

    mach_Log_Classifier_MLJ_PUY_L2=machine(self_Log_Classifier_MLJ_PUY, select(train_PUY, Not(:precipitation_nextday)),train_PUY.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_MLJ_PUY_L2)
end
begin
    evaluate!(mach_Log_Classifier_PUY_L2, measure=auc)
    exportMachine("mach_Log_Classifier_MLJ_PUY_L2.jlso", mach_Log_Classifier_MLJ_PUY_L2)
end

##LogisticClassifier (MLJ) filled data ALL L2
begin
    mach_Log_Classifier_MLJ_Filled_L2=machine(self_Log_Classifier_MLJ_L2, select(train_filled, Not(:precipitation_nextday)),train_filled.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_MLJ_Filled_L2).best_model
end
begin
    exportMachine("mach_Log_Classifier_MLJ_Filled_L2.jlso", mach_Log_Classifier_MLJ_Filled_L2)
end

## LogisticClassifier (MLJ) standerdized data ALL No Regularisation
begin
    train_std_mach=machine(Standardizer(), select(train, Not(:precipitation_nextday)))|>fit!
    train_std= MLJ.transform(train_std_mach, train)
    mach_Log_Classifier_MLJ_STD=machine(LogisticClassifier(penalty=:none), select(train_std, Not(:precipitation_nextday)),train_std.precipitation_nextday)|> fit!
    evaluate!(mach_Log_Classifier_MLJ_STD, measure=auc)
end
begin 
    exportMachine("mach_Log_Classifier_STD_MLJ_ALL.jlso", mach_Log_Classifier_MLJ_STD)
end

#--------------------------------------------------------------
#Stochstic gradient descent on LR -->not workig for the moment
begin
    using MLJScikitLearnInterface
    model_LR_GD=SGDClassifier(prediction_type=:deterministic)
    prediction_type(model_LR_GD)=:probabilistic
    mach_LR_GD=machine(model_LR_GD, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|>MLJ.fit!
    summary(mach_LR_GD)
    evaluate!(mach_LR_GD, measure=auc)
    predict(mach_LR_GD, select(train, Not(:precipitation_nextday)))
end

