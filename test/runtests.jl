using CompNMF
using Test

@testset "CompNMF.jl" begin
    for T in (Float64, Float32)
        W, H = zeros(T,6,3), zeros(T,3,6)
        W[1:2,1] .= ones(T,2); W[3:4,2] .= ones(T,2); W[5:6,3] .= ones(T,2)
        H[1,1:2] .= ones(T,2); H[2,3:4] .= ones(T,2); H[3,5:6] .= ones(T,2)
        X = W*H; X = X .+ rand(T, size(X)...)*T(0.1)
        CompNMF.solve!(CompNMF.CompressedNMF{T}(maxiter=1000, tol=1e-9), X, W, H)
        @test X ≈ W * H atol=1e-4
    end
end
