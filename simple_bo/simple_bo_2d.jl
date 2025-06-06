using Measures
using Distributions
using GaussianProcesses
using Plots

include("helper_fncs.jl")
BUDGET = 10 # Number of iterations that will be done with this implementation of BO

# Gets the objective function so we can check our values at the end
f_obj = create_objective_function()
x = range(-0.5,3, length = 100)
y = f_obj.(x)
y_max = findmax(y)
# p1 = plot(x, y, label = "Objective function")

x_samp = range(-0.5, 3, length = 3)
x_init = collect(x_samp)
y_init = f_obj.(x_init) 
𝒟_ei = (x_init, y_init) # Observed Data points
𝒟_ucb = (x_init, y_init) # Observed Data points

m = MeanZero()
squared_exponential = SE(.01, 3.0)
GP_ei = GP(x_init, y_init, m, squared_exponential)
GP_ucb = GP(x_init, y_init, m, squared_exponential)
plots = []
X = filter(x_rm -> !(x_rm in x_init), x)

function main(X, 𝒟_ei, 𝒟_ucb, GP_ei, GP_ucb, f_obj, squared_exponential, m, plots, BUDGET)    
    for i = 1:BUDGET   
        show_legend = i == 1
        exp_imp = expected_improvement(X, 𝒟_ei, GP_ei)
        p_acq_ei = plot(X, exp_imp, label = show_legend ? "Expected Improvement Acquisition" : "", left_margin = 35mm, legend = show_legend ? :topright : false)
        𝒟_ei = best_sampling_point(exp_imp, GP_ei, X, 𝒟_ei, f_obj)
        GP_ei = GP(𝒟_ei[1], 𝒟_ei[2], m, squared_exponential)
        p_gp_ei = plot(GP_ei; label = show_legend ? "Expected Improvement Surrogate Function" : "", left_margin = 30mm,  top_margin = 10mm, title = "Iteration $i", legend = show_legend, obsv = false)
        plot!(p_gp_ei, x, y, label = show_legend ? "Objective Function" : "", legend = show_legend)
        scatter!(p_gp_ei, 𝒟_ei[1], 𝒟_ei[2], label = show_legend ? "EI Samples" : "", 
            color=:black, marker=:xcross, markersize=5, markerstrokewidth=:1, legend = show_legend)
        
        ucb = upper_confidence_bounds(X, GP_ucb)
        p_acq_ucb = plot(X, ucb, label = show_legend ? "Upper Confidence Bound Acquisition" : "", legend = show_legend ? :topright : false)
        𝒟_ucb = best_sampling_point(ucb, GP_ucb, X, 𝒟_ucb, f_obj)
        GP_ucb = GP(𝒟_ucb[1], 𝒟_ucb[2], m, squared_exponential)
        p_gp_ucb = plot(GP_ucb; label = show_legend ? "Upper Confidence Bound Surrogate Function" : "", left_margin = 30mm,  top_margin = 10mm, title = "Iteration $i", legend = show_legend, obsv = false)
        plot!(p_gp_ucb, x, y, label = show_legend ? "Objective Function" : "", legend = show_legend)
        scatter!(p_gp_ucb, 𝒟_ucb[1], 𝒟_ucb[2], label = show_legend ? "UCB Samples" : "", 
            color=:red, marker=:xcross, markersize=5, markerstrokewidth=:1, legend = show_legend)
        
            
        push!(plots, [p_gp_ei, p_acq_ei, p_gp_ucb, p_acq_ucb])
    end
    all_plots = reduce(vcat, plots)
    display(plot(all_plots..., layout = (BUDGET, 4), size = (6000, 8000)))
    max_ei = findmax(𝒟_ei[2])
    max_ucb = findmax(𝒟_ucb[2])
    return (max_ei[1], max_ucb[1])
end

dataset_max = main(X, 𝒟_ei, 𝒟_ucb, GP_ei, GP_ucb, f_obj, squared_exponential, m, plots, BUDGET) 
println("Objective function maximum value: $(y_max[1])\nExpected improvement surrogate function maximum value: $(dataset_max[1])")
error_perc = abs(dataset_max[1] - y_max[1]) / y_max[1] * 100
println("Error calculation: $error_perc", "%")
println("Objective function maximum value: $(y_max[1])\nUpper confidence bound surrogate function maximum value: $(dataset_max[2])")
error_perc = abs(dataset_max[2] - y_max[1]) / y_max[1] * 100
println("Error calculation: $error_perc", "%")

