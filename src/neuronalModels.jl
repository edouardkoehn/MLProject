### Main file for optimising neuronal network

##Load the daata
begin
    using CSV, DataFrames, Plots,MLJ,MLJFlux,Random,NNlib, Flux
    include("utils.jl")

    #import test
    test=importTest()
    #import cleaned train
    train=importTrainCleaned()
    #import filled train
    train_filled=importTrainFilled()
    #subset for PUY weather station  
    train_PUY= hcat(select(train, r"PUY"), DataFrame(precipitation_nextday=train.precipitation_nextday))
end

## Neuronal tuning process
begin
    model_NN = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 200,σ = relu, dropout=0.1),
                                    optimiser = ADAM(),
                                    batch_size = 32)

    
    self_NN = TunedModel(model = model_NN,
                       resampling = CV(nfolds = 5),
                       range = [range(model_NN,
                                 :(builder.dropout),
                                 values = [0., .1, .2]),
                                range(model_NN,
                                  :epochs,
                                  values = [500, 1000, 2000])],
                                  measure=auc)

    train_std=MLJ.transform(machine(Standardizer(), select(train, Not(:precipitation_nextday)))|>fit!,train)
    
    mach_NN_1=MLJ.fit!(machine(self_NN, select(train, Not(:precipitation_nextday)), train.precipitation_nextday))
    
end
begin
    report(mach_NN_1)
    plotMachine(mach_NN_1,"NN")
    exportMachine("mach_NN_Classifier_n=200.jlso", mach_NN_1)
    evaluate!(mach_NN_1, measure=auc)
end

## Neuronal network optimised
begin
    model_NN_simple=NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 125,
                                                            dropout = 0.2,
                                                            σ = relu),
                                        optimiser = ADAM(0.001, (0.9, 0.999), IdDict{Any,Any}()),
                                        loss = Flux.crossentropy,
                                        batch_size = 500,
                                        lambda = 0.0,
                                        alpha = 0.0,
                                        epochs= 1000,
                                        optimiser_changes_trigger_retraining = false)
    mach_NN_2=machine(model_NN_simple,select(train, Not(:precipitation_nextday)), train.precipitation_nextday)|>fit!                   
end
begin
    evaluate!(mach_NN_2,measure=auc)
    exportMachine("mach_NN_optimised.jlso", mach_NN_2)
end


