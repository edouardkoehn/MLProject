using MLJ, CSV, NearestNeighborModels,MLJFlux,Flux,MLJXGBoostInterface

function exportMachine(fileName,model ) 
    MLJ.save(joinpath(@__DIR__, "..", "models", fileName),model)
end

function loadMachine(fileName)
   return machine(joinpath(@__DIR__, "..", "models", fileName))
end

function plotMachine(machine,name)
    rep=report(machine)
    scatter(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements,label = rep.plotting.parameter_names[1], ylabel = "AUC", title=name)
end

function writeSubmission(test, machine, filename)
    sub=predict(machine, test).prob_given_ref
    df=DataFrame(id=1:1200,precipitation_nextday=sub[2])
    CSV.write(joinpath(@__DIR__, "..", "submissions", filename),df)
    return df
end

function cross_validation_sets(idx, K=10)
    n = length(idx)
    r = n ÷ K
    [let idx_valid = idx[(i-1)*r+1:(i == K ? n : i*r)]
         (idx_valid = idx_valid)
     end
     for i in 1:K]
end

function CV_Auc(machine, data,K)
    auc=[]
    idx=cross_validation_sets(1:size(data)[1],K)
    for i in idx
        append!(auc, MLJ.auc(predict(machine, select(data[i,:], Not(:precipitation_nextday))), data.precipitation_nextday[i]))
    end
    return auc
end
