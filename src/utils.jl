using MLJ, CSV, NearestNeighborModels,MLJFlux,Flux,MLJXGBoostInterface

function exportMachine(fileName,model )
    #Methods for saving tained machine 
    MLJ.save(joinpath(@__DIR__, "..", "models", fileName),model)
end

function loadMachine(fileName)
    #Methods for loading trained machine
   return machine(joinpath(@__DIR__, "..", "models", fileName))
end

function plotMachine(machine,name)
    #Methods for ploting self tuned model report
    rep=report(machine)
    scatter(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements,label = rep.plotting.parameter_names[1], ylabel = "AUC", title=name)
end

function writeSubmission(test, machine, filename)
    #Methods for producing the submision with a trained model
    sub=predict(machine, test).prob_given_ref
    df=DataFrame(id=1:1200,precipitation_nextday=sub[2])
    CSV.write(joinpath(@__DIR__, "..", "submissions", filename),df)
    return df
end


function importTest()
    #Function for importing the test data
    test=CSV.read(joinpath(@__DIR__, "..", "data", "testdata.csv"), DataFrame)
    println("Shape of the test set :",size(test))
    dropmissing!(test)
    println("Shape of the cleaned test set :",size(test))
    return test
end

function importTrainCleaned()
    #Methods for importing the cleaned data
    train=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    coerce!(train,:precipitation_nextday => Multiclass)
    dropmissing!(train)
    train=train[shuffle(1:size(train, 1)),:]
    println("Shape of the cleaned training set :" ,size(train))
    return train
end

function importTrainFilled()
    #Methods for importing the cleaned data
    train=CSV.read(joinpath(@__DIR__, "..", "data", "trainingdata.csv"), DataFrame)
    coerce!(train,:precipitation_nextday => Multiclass)
    train_filled= MLJ.transform(fit!(machine(FillImputer(), train)), train)
    coerce!(train_filled,:precipitation_nextday => Multiclass)
    train_filled=train_filled[shuffle(1:size(train_filled, 1)),:]

    println("Shape of the training set filled :" ,size(train_filled))
    return train_filled
end