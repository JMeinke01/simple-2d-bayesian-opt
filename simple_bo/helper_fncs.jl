using Distributions
using GaussianProcesses
using LaTeXStrings

#Defining our objective function to check accuracy of guess later:
function create_objective_function() 
        a, b, c = [rand(Uniform(0.1,1)) for _ in 1:3]
        d = rand(Uniform(0.05, 0.2))
        e = rand(Uniform(0.5,1.5))
        return x -> @. a * sin(3 * x) + 
        b * sin(5 * x) + 
        c * cos(7 * x) + 
        d * x^2 - 
        e * x
end

#Given a data set of sample points, compute the expected improvement at points X
function expected_improvement(X, 𝒟, GP; ζ = 0.05)
        μ,Σ = predict_f(GP, X, full_cov = true)
        σ = sqrt.(diag(Σ)) 
        σ = reshape(σ, :, 1)
        f_opt = maximum(𝒟[2])

        imp = @. (μ - f_opt - ζ)
        z = imp ./ σ
        exp_imp =  imp .* cdf.(Ref(Normal()), z) .+ σ .* pdf.(Ref(Normal()), z)
        return exp_imp   
end   

# Given a data set of sample points, attempts to maximize the objective function by calculating the trade-off between exploration vs. exploitation
function upper_confidence_bounds(X, GP; ζ = 25.0) # This value of ζ is more geared toward exploration
        μ,Σ = predict_f(GP, X, full_cov = true)
        σ = sqrt.(diag(Σ)) 
        σ = reshape(σ, :, 1)
        return (μ .+ ζ .* σ)
end   

#Given an acquisition function, propose the next best sampling location
function best_sampling_point(acq_func, GP, X, 𝒟, f_obj) 
        samp_pt = findmax(acq_func)
        new_x = X[samp_pt[2]]
        new_y = f_obj(new_x)
        x_new = vcat(𝒟[1], new_x)
        y_new = vcat(𝒟[2], new_y)
        return (x_new, y_new)
end
