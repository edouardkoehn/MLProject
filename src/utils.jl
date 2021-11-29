using MLJ, CSV, NearestNeighborModels
function exportMachine(fileName,model ) 
    MLJ.save(joinpath(@__DIR__, "..", "models", fileName),model)
end

function loadMachine(fileName)
   return machine(joinpath(@__DIR__, "..", "models", fileName))
end

function plotMachine(machine,name)
    rep=report(machine)
    scatter(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements, xlabel = rep.plotting.parameter_names[1], ylabel = "AUC", title=name)
end

function writeSubmission(test, machine, filename)
    sub=predict(machine, test).prob_given_ref
    df=DataFrame(id=1:1200,precipitation_nextday=sub[2])
    CSV.write(joinpath(@__DIR__, "..", "submissions", filename),df)
    return df
end