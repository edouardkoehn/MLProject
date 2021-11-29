#Main file for testing the different linear MLJLinearModels
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    include("utils.jl")
end
#------------------------------------------------------------------
# Import of the data test
begin
    using CSV, DataFrames, Plots,MLJ,MLJLinearModels,StatsPlots
    test=CSV.read(joinpath(@__DIR__, "..", "data", "testdata.csv"), DataFrame)
    dropmissing!(test)
  
    train=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    dropmissing!(train)
    coerce!(train,:precipitation_nextday => Multiclass)
    train=train[shuffle(1:size(train, 1)),:]

    train_PUY= hcat(select(train, r"PUY"), DataFrame(precipitation_nextday=train.precipitation_nextday))
end
#------------------------------------------------------------------
#LogisticClassifier without restictrion

begin
    mach_Log_Classifier_MLJ_NoR= machine(LogisticClassifier(penalty=:none), select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
    fitted_params(mach_Log_Classifier_MLJ_NoR)
end
#------------------------------------------------------------------
#LogisticClassifier (MLJ) All
begin
    Log_Classifier_MLJ= LogisticClassifier()
    self_Log_Classifier_MLJ=TunedModel(model = Log_Classifier_MLJ,
        resampling = CV(nfolds = 10),
        tuning = Grid(),
        range=range(Log_Classifier_MLJ, :lambda,
        lower = 1e-5, upper = 3e2,
        scale = :log),
        measure = auc)
    mach_Log_Classifier_MLJ=machine(self_Log_Classifier_MLJ, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_MLJ)
end
begin
        exportMachine("mach_Log_Classifier_MLJ_ALL.jlso", mach_Log_Classifier_MLJ) 
end
#------------------------------------------------------------------
#LogisticClassifier (MLJ) PUY
begin
    mach_Log_Classifier_MLJ_PUY=machine(self_Log_Classifier_MLJ, select(train_PUY, Not(:precipitation_nextday)),train_PUY.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_MLJ_PUY).best_model
end
begin
    exportMachine("mach_Log_Classifier_MLJ_PUY.jlso", mach_Log_Classifier_MLJ_PUY)
end
#------------------------------------------------------------------
#LogisticClassifier (MLJ) PUY L2 same results as the classic LogisticClassifier

#------------------------------------------------------------------
#KNNClassifier (MLJ) PUY
begin
    using NearestNeighborModels
    KNN_Classifier=KNNClassifier()
    self_KNN_Classifier=TunedModel(model = KNN_Classifier,
            resampling = CV(nfolds = 10),
            tuning = Grid(),
            range=range(KNN_Classifier, :K, values = 1:70),
            measure = auc)
        mach_KNN_Classifier_PUY=machine(self_KNN_Classifier, select(train_PUY, Not(:precipitation_nextday)),train_PUY.precipitation_nextday)|> fit!
        report(mach_KNN_Classifier_PUY).best_model
end
begin
    
    exportMachine("mach_KNN_Classifier_MLJ_PUY.jlso", mach_KNN_Classifier_PUY)
end

#------------------------------------------------------------------
#KNNClassifier (MLJ) ALL
begin
        mach_KNN_Classifier_ALL=machine(self_KNN_Classifier, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
        report(mach_KNN_Classifier_ALL).best_model
end
begin
    exportMachine("mach_KNN_Classifier_MLJ_ALL.jlso", mach_KNN_Classifier_ALL)
end

#------------------------------------------------------------------
#Mutlinomial classifier --> really bad results
#--------------------------------------------------------------
#LogisticClassifier (MLJ) standerdized -> performing badly
begin
    train_std_mach=fit!(machine(Standardizer(), select(train, Not(:precipitation_nextday))));
    train_std= MLJ.transform(train_std_mach, train)
    mach_Log_Classifier_Std_MLJ_ALL=machine(self_Log_Classifier_MLJ, select(train_std, Not(:precipitation_nextday)),train_std.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_Std_MLJ_ALL).best_model
end
begin 
    exportMachine("mach_Log_Classifier_STD_MLJ_ALL.jlso", mach_Log_Classifier_Std_MLJ_ALL)
end

#--------------------------------------------------------------
#KNNClassifier (MLJ) PUY standerdized -> performing badly (not working for the moment)
begin
    mach_KNN_Classifier_ALL_Std=machine(self_KNN_Classifier, select(train_std, Not(:precipitation_nextday)),train_std.precipitation_nextday)|> fit!
    report(mach_KNN_Classifier_ALL_Std)
end
begin 
    report(mach_KNN_Classifier_ALL_Std).best_model
    exportMachine("mach_KNN_Classifier_STD_MLJ_ALL.jlso", mach_KNN_Classifier_ALL_Std)
end
#--------------------------------------------------------------
#RidgleClassifier --> not working
begin
    using MLJScikitLearnInterface
    Ridge_Classifier =  RidgeClassifier() 
    self_Ridge_Classifier=TunedModel(model = Ridge_Classifier,
            resampling = CV(nfolds = 5),
            tuning = Grid(),
            range=range(Ridge_Classifier, :alpha,values = 1:100),
            measure = auc)
        mach_Ridge_Classifier=machine(self_Ridge_Classifier, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> MLJ.fit!
        report(mach_Ridge_Classifier)
end

#--------------------------------------------------------------
#AdaBoostStumpClassifier
begin
    using DecisionTree
    AdaBoostStump_Classifier =  AdaBoostStumpClassifier(n_iterations=100) 
    
    mach_AdaBoostStump_Classifier=machine(AdaBoostStump_Classifier, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
        report(mach_Ridge_Classifier)
end
