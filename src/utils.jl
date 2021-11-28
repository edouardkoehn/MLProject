using MLJ, CSV
function exportMachine(fileName,model ) 
    MLJ.save(joinpath(@__DIR__, "..", "models", fileName),model)
end

function loadMachine(fileName)
   return machine(joinpath(@__DIR__, "..", "models", fileName))
end

function plotMachine(machine)
    rep=report(machine)
    scatter!(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements, xlabel = rep.plotting.parameter_names[1], ylabel = "AUC")
end

function writeSubmission(test, machine, filename)
    sub=predict(machine, test)
    CSV.write(joinpath(@__DIR__, "..", "submissions", fileName),sub )
end