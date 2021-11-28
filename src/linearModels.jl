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
end
#------------------------------------------------------------------
#LogisticClassifier (MLJ) All
begin
    Log_Classifier_MLJ= LogisticClassifier()
    self_Log_Classifier_MLJ=TunedModel(model = Log_Classifier_MLJ,
        resampling = CV(nfolds = 10),
        tuning = Grid(),
        range=range(Log_Classifier_MLJ, :lambda,
        lower = 1e-3, upper = 2e2,
        scale = :log),
        measure = auc)
    mach_Log_Classifier_MLJ=machine(self_Log_Classifier_MLJ, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_MLJ)
end
begin
        MLJ.save("models/mach_Log_Classifier_MLJ.jlso", mach_Log_Classifier_MLJ) 
end
#------------------------------------------------------------------
#LogisticClassifier (MLJ) PUY
begin
    mach_Log_Classifier_MLJ_PUY=machine(self_Log_Classifier_MLJ, select(train,r"PUY", Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_MLJ_PUY)
end
begin
    MLJ.save("models/mach_Log_Classifier_MLJ_PUY.jlso", mach_Log_Classifier_MLJ_PUY)
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
            range=range(KNN_Classifier, :K, values = 50:200),
            measure = auc)
        mach_KNN_Classifier_PUY=machine(self_KNN_Classifier, select(train,r"PUY", Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
        report(mach_KNN_Classifier_PUY)
end
begin
    
    MLJ.save("models/mach_KNN_Classifier_MLJ_PUY.jlso", mach_KNN_Classifier_PUY)
end

#------------------------------------------------------------------
#KNNClassifier (MLJ) ALL
begin
        mach_KNN_Classifier_ALL=machine(self_KNN_Classifier, select(train, Not(:precipitation_nextday)),train.precipitation_nextday)|> fit!
        report(mach_KNN_Classifier_ALL)
end
begin
    MLJ.save("models/mach_KNN_Classifier_MLJ_ALL.jlso", mach_KNN_Classifier_ALL)
end

#------------------------------------------------------------------
#Mutlinomial classifier --> really bad results
#--------------------------------------------------------------
#LogisticClassifier (MLJ) PUY standerdized -> performing badly
begin
    train_std_mach=fit!(machine(Standardizer(), select(train, Not(:precipitation_nextday))));
    train_std= MLJ.transform(train_std_mach, train)
    mach_Log_Classifier_Std_MLJ_ALL=machine(self_Log_Classifier_MLJ, select(train_std, Not(:precipitation_nextday)),train_std.precipitation_nextday)|> fit!
    report(mach_Log_Classifier_Std_MLJ_ALL)
end
begin 
    MLJ.save("models/mach_Log_Classifier_STD_MLJ_ALL.jlso", mach_Log_Classifier_Std_MLJ_ALL)
end

#--------------------------------------------------------------
#KNNClassifier (MLJ) PUY standerdized -> performing badly
begin
    mach_KNN_Classifier_Std=machine(self_KNN_Classifier, select(train_std, Not(:precipitation_nextday)),train_std.precipitation_nextday)|> fit!
    report(mach_KNN_Classifier_Std)
end
begin 
    MLJ.save("models/mach_KNN_Classifier_STD_MLJ_ALL.jlso", mach_KNN_Classifier_Std)
end
#--------------------------------------------------------------
#plotting KNN
begin
    scatter(reshape(report(mach_KNN_Classifier_ALL).plotting.parameter_values, :),
    report(mach_KNN_Classifier_ALL).plotting.measurements, xlabel = report(mach_KNN_Classifier_ALL).plotting.parameter_names[1], ylabel = "AUC", lable="KNN All")

    scatter!(reshape(report(mach_KNN_Classifier).plotting.parameter_values, :),
    report(mach_KNN_Classifier).plotting.measurements, xlabel = report(mach_KNN_Classifier).plotting.parameter_names[1], ylabel = "AUC", label="KNN PUY")
end

begin
    scatter(reshape(report(mach_Log_Classifier_MLJ).plotting.parameter_values, :),
    report(mach_Log_Classifier_MLJ).plotting.measurements, xlabel = report(mach_Log_Classifier_MLJ).plotting.parameter_names[1], ylabel = "AUC", label="Log All")

    scatter!(reshape(report(mach_Log_Classifier_MLJ_PUY).plotting.parameter_values, :),
    report(mach_Log_Classifier_MLJ_PUY).plotting.measurements, xlabel = report(mach_Log_Classifier_MLJ_PUY).plotting.parameter_names[1], ylabel = "AUC", label="Log PUY")

    
    scatter!(reshape(report(mach_Log_Classifier_Std_MLJ_PUY).plotting.parameter_values, :),
    report(mach_Log_Classifier_Std_MLJ_PUY).plotting.measurements, xlabel = report(mach_Log_Classifier_Std_MLJ_PUY).plotting.parameter_names[1], ylabel = "AUC", label="Log PUY")
end

