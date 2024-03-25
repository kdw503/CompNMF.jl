module CompNMF

using LinearAlgebra, NMF, RandomizedLinAlg, DataStructures, StatsBase
export solve!, CompressedNMF, compmat

include("utils.jl")

mutable struct CompressedNMF{T}
    maxiter::Int           # maximum number of iterations (in main procedure)
    verbose::Bool          # whether to show procedural information
    tol::T                 # tolerance of changes on U and V upon convergence
    xi::T
    lambda::T
    phi::T
    SCA_penmetric::Symbol  # SCA add. :HALS, :SCA
    SCA_αw::T
    SCA_αh::T

    function CompressedNMF{T}(;maxiter::Integer=100,
                              verbose::Bool=false,
                              tol::Real=cbrt(eps(T)),
                              xi::Real=1.0,
                              lambda::Real=1.0,
                              phi::Real=1.0,
                              SCA_penmetric::Symbol=:CompNMF,
                              SCA_αw::Real=100,
                              SCA_αh::Real=100) where T
        new{T}(maxiter, verbose, tol, xi, lambda, phi, SCA_penmetric, SCA_αw, SCA_αh)
    end
end

mutable struct CompressedNMFState{T}
    L::Matrix{T}
    R::Matrix{T}
    A_tilde::Matrix{T}
    gtU::Matrix{T}
    gtV::Matrix{T}
    normU::T
    normV::T
    X_tilde::Matrix{T}
    Y_tilde::Matrix{T}
    Lambda::Matrix{T}
    Phi::Matrix{T}
    TmpUs::Vector{Matrix{T}}
    TmpVs::Vector{Matrix{T}}
    TmpXts::Vector{SubArray{T}}
    TmpYts::Vector{SubArray{T}}
    TmpMs::Vector{SubArray{T}}
    function CompressedNMFState{T}(X_tilde, Y_tilde, L, R, A_tilde, gtU, gtV) where T
        m, rrov = size(L); rrov, n = size(R); r = size(X_tilde,2)
        Lambda, Phi = zeros(T,m,r), zeros(T,r,n)
        TmpUs = map(i->Matrix{T}(undef,m,r),1:2)
        TmpVs = map(i->Matrix{T}(undef,r,n),1:2)
        Ms = map(i->Matrix{T}(undef,rrov,rrov),1:2)
        TmpXts = map(i->view(Ms[i],1:rrov,1:r),1:2)
        TmpYts = map(i->view(Ms[i],1:r,1:rrov),1:2)
        TmpMs = map(i->view(Ms[i],1:r,1:r),1:2)
        new{T}(L, R, A_tilde, gtU, gtV, 0., 0., X_tilde, Y_tilde,
               Lambda, Phi, TmpUs, TmpVs, TmpXts, TmpYts, TmpMs)
    end
end

struct CompressedNMFUpd{T} <: NMF.NMFUpdater{T}
    xi::T
    lambda::T
    phi::T
    SCA_penmetric::Symbol
    SCA_αw::T
    SCA_αh::T
    function CompressedNMFUpd{T}(xi::T,lambda::T,phi::T,SCA_penmetric::Symbol,SCA_αw::T, SCA_αh::T) where {T}
        new{T}(xi, lambda, phi, SCA_penmetric, SCA_αw, SCA_αh)
    end
end

struct Result{T}
    L::Matrix{T}
    R::Matrix{T}
    A_tilde::Matrix{T}
    X_tilde::Matrix{T}
    Y_tilde::Matrix{T}
    niters::Int
    converged::Bool
    objvalue::T
    objvalues::Vector{T}
    sparsevalues::Vector{T}
    avgfits::Vector{T}
    inittime::T
    function Result{T}(L::Matrix{T}, R::Matrix{T}, A_tilde::Matrix{T}, X_tilde::Matrix{T}, Y_tilde::Matrix{T},
            niters::Int, converged::Bool, objv, objvs, sparsevalues, avgfits, inittime) where T
       new{T}(L, R, A_tilde, X_tilde, Y_tilde, niters, converged, objv, objvs, sparsevalues, avgfits, inittime)
    end
end

function prepare_state(::CompressedNMFUpd{T}, A, U, V; L=nothing, R=nothing, gtU::Matrix{T}=Matrix{T}(undef,0,0),
        gtV::Matrix{T}=Matrix{T}(undef,0,0)) where T
    time0 = time()
    if (L === nothing) || (R === nothing)
        L, R, X_tilde, Y_tilde, A_tilde = compmat(A, U, V; w=4)
    else
        A_tilde = L'*A*R'
        X_tilde, Y_tilde = L'U, V*R'
    end
    inittime = time()-time0
    state = CompressedNMFState{T}(X_tilde, Y_tilde, L, R, A_tilde, gtU, gtV)
    state, inittime # U and V are not stored
end
function double_op_nlv!(fn::Function,C,A,B)
    @inbounds @simd for i in eachindex(A)
        C[i] = fn(A[i],B[i])
    end
    return C
end
function double_op_nlv!(fn::Function,C,A,b::Real)
    @inbounds @simd for i in eachindex(A)
        C[i] = fn(A[i],b)
    end
    return C
end
sprod!(C,A,b) = double_op_nlv!(*,C,A,b)
sdiv!(C,A,b) = double_op_nlv!(/,C,A,b)
madd!(C,A,B) = double_op_nlv!(+,C,A,B)
msub!(C,A,B) = double_op_nlv!(-,C,A,B)
function mnonneg!(A)
    @inbounds @simd for i in eachindex(A)
        A[i] = max(A[i],0.)
    end
    return A
end

"""
low rank matrix appoximation
w = 4 in "Compressed Nonnegative Matrix Factorization Is Fast and Accurate (2016)"
w = 1 or 2 in "Finding structure with randomness: probabilistic algorithms for
                constructing approximate matrix decompositions (2010)"
"""
function low_rank_QR(A::AbstractArray{T,2}, rrov; w=4) where T<:Real
    Ω = randn(T,size(A,2),rrov)
    AAtw = (A*A')^w; AΩ = A*Ω
    Q, _ = qr(AAtw*AΩ)
    Matrix(Q)
end

function compmat(A::AbstractArray{T,2}, Up, Vp; w=4, rov=10) where T
    r = size(Up,2)
    L = low_rank_QR(A,r+rov,w=w)
    R = Array(low_rank_QR(A',r+rov,w=w)')
    A_tilde = L'*A*R'
    # balanceUV!(Up,Vp)
    X_tilde, Y_tilde = L'Up, Vp*R'
    # @show norm(A-L*L'A*R'R)^2
    L, R, X_tilde, Y_tilde, A_tilde
end

function balanceUV!(Un, Vn)
    for k in 1:size(Un,2)
        balanceUkVk!(view(Un,:,k), view(Vn,k,:))
    end
    Un, Vn
end
function balanceUkVk!(Uk, Vk)
    normw = max(eps(eltype(Uk)),norm(Uk))
    normh = max(eps(eltype(Vk)),norm(Vk))
    balfacs = sqrt(normw/normh)
    Uk ./= balfacs; Vk .*= balfacs
end

solve!(alg::CompressedNMF{T}, A, U, V; L=nothing,R=nothing,
        gtU::Matrix{T}=Matrix{T}(undef,0,0), gtV::Matrix{T}=Matrix{T}(undef,0,0),
        maskU::Union{Colon,Vector,BitVector}=Colon(),maskV::Union{Colon,Vector,BitVector}=Colon()) where {T} =
    nmf_skeleton!(CompressedNMFUpd{T}(alg.xi, alg.lambda, alg.phi, alg.SCA_penmetric, alg.SCA_αw, alg.SCA_αh),
            A, U, V, alg.maxiter, alg.verbose, alg.tol; L=L, R=R, gtU=gtU, gtV=gtV, maskU=maskU, maskV=maskV)

function evaluate_objv(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, V) where T
    # convert(T, 0.5) * sqL2dist(A, s.UV)
    if updater.SCA_penmetric ∈ [:HALS, :SPARSE_U, :SPARSE_V]
        norm(A-U*V)^2
    elseif updater.SCA_penmetric == :SCA
        X_tilde = s.L'*U; Y_tilde = V*s.R'
        norm(Diagonal(s.A_tilde)-X_tilde*Y_tilde)^2
    elseif updater.SCA_penmetric == :CompNMF
        p1 = norm(s.A_tilde-s.X_tilde*s.Y_tilde)^2
        p2 = sum(s.Lambda.*(s.L*s.X_tilde-U))
        p3 = updater.lambda/2*norm(s.L*s.X_tilde-U)^2
        p4 = sum(s.Phi.*(s.Y_tilde*s.R-V))
        p5 = updater.phi/2*norm(s.Y_tilde*s.R-V)^2
        p1 + p2 + p3 + p4 + p5
    end
end
function evaluate_sparseness(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, V) where T
    if updater.SCA_penmetric == :SCA
        X_tilde = s.L'*U; Y_tilde = V*s.R'
        normL1 = norm(s.L,1); normR1 = norm(s.R,1); (αw, αh) = (updater.SCA_αw/normL1, updater.SCA_αh/normR1)
        αw*norm(s.L*X_tilde,1) + αh*norm(Y_tilde*s.R,1)
    elseif updater.SCA_penmetric == :SPARSE_U
        Un, Vn = copy(U), copy(V); normalizeU!(Un,Vn)
        norm(Un,1)#/s.normU
    elseif updater.SCA_penmetric == :SPARSE_V
        norm(V,1)#/s.normV
    else
        zero(T)
    end
end
function evaluate_fitvalue(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, V, maskU, maskV) where T
    if !isempty(s.gtU) && !isempty(s.gtV)
#        U = s.L*s.X_tilde; V = s.Y_tilde*s.R  # this is for U=L*X, V=Y*R output instead of just U, V
        avgfit, _ =  matchedfitval(s.gtU, s.gtV, U[maskU,:], V[:,maskV]; clamp=false)
    else
        avgfit = fitd(A[maskU,maskV], U[maskU,:]*V[:,maskV])
    end
    avgfit
end

function nmf_skeleton!(updater::NMF.NMFUpdater{T},
                       A, U::Matrix{T}, V::Matrix{T},
                       maxiter::Int, verbose::Bool, tol;
                       L=nothing, R=nothing,
                       gtU::Matrix{T}=Matrix{T}(undef,0,0),
                       gtV::Matrix{T}=Matrix{T}(undef,0,0),
                       maskU::Union{Colon,Vector,BitVector}=Colon(),
                       maskV::Union{Colon,Vector,BitVector}=Colon()
                       ) where T
    objv = convert(T, NaN)
    # init

    state, inittime = prepare_state(updater, A, U, V; L=L, R=R, gtU=gtU[maskU,:], gtV=gtV[maskV,:])
    preU = Matrix{T}(undef, size(U))
    preV = Matrix{T}(undef, size(V))
    objvs = T[]; objvsparses = T[]; avgfits=T[]
    if verbose
        start = time()
        objv = evaluate_objv(updater, state, A, U, V)
        push!(objvs,objv)
        push!(objvsparses,evaluate_sparseness(updater, state, A, U, V))
        push!(avgfits,evaluate_fitvalue(updater, state, A, U, V, maskU, maskV))
        # @printf("%-5s    %-13s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change", "(U & V).change")
        # @printf("%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
    end

    # main loop
    converged = false
    iter = 0
    while !converged && iter < maxiter
        iter += 1
        copyto!(preU, U)
        copyto!(preV, V)

        # update V
        update_wh!(updater, state, A, U, V)

        # determine convergence
        dev = max(maxad(preU, U), maxad(preV, V))
        if dev < tol
            converged = true
        end

        # display info
        if verbose
            elapsed = time() - start
            preobjv = objv
            objv = evaluate_objv(updater, state, A, U, V)
            push!(objvs,objv)
            push!(objvsparses,evaluate_sparseness(updater, state, A, U, V))
            push!(avgfits,evaluate_fitvalue(updater, state, A, U, V, maskU, maskV))
            #@printf("%5d    %13.6e    %13.6e    %13.6e    %13.6e\n",
            #    t, elapsed, objv, objv - preobjv, dev)
        end
    end
    if !verbose
        objv = evaluate_objv(updater, state, A, U, V)
    end
 #   return Result{T}(U, V, iter, converged, objv, objvs, objvsparses, avgfits, inittime)
    return Result{T}(state.L, state.R, state.A_tilde, state.X_tilde, state.Y_tilde, iter,
                    converged, objv, objvs, objvsparses, avgfits, inittime)
end

function update_wh!(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, V) where T
    Us = s.TmpUs; Vs = s.TmpVs; Xts = s.TmpXts; Yts = s.TmpYts; Ms = s.TmpMs
    xi = updater.xi; lambda = updater.lambda; phi = updater.phi
    xilambda = xi*lambda; xiphi = xi*phi

    # s.X_tilde .= (s.A_tilde*s.Y_tilde'+lambda*s.L'U-s.L's.Lambda)*inv(s.Y_tilde*s.Y_tilde'+lambda*Matrix(1.0I,r,r))
    mul!(Xts[1],s.A_tilde,s.Y_tilde')
    mul!(Xts[2],s.L',U); rmul!(Xts[2],lambda)
    madd!(Xts[1],Xts[1],Xts[2])
    mul!(Xts[2],s.L',s.Lambda); msub!(Xts[1],Xts[1],Xts[2])
    mul!(Ms[2], s.Y_tilde, s.Y_tilde'); Ms[2][diagind(Ms[2])] .+= lambda
    try
        mul!(s.X_tilde,Xts[1],inv(Ms[2]))
    catch e
        @show Ms[2]
        error(e)
    end

    # s.Y_tilde .= inv(s.X_tilde's.X_tilde+phi*Matrix(1.0I,r,r))*(s.X_tilde's.A_tilde+phi*V*s.R'-s.Phi*s.R')
    mul!(Yts[1],s.X_tilde',s.A_tilde)
    mul!(Yts[2],V,s.R'); rmul!(Yts[2],phi)
    madd!(Yts[1],Yts[1],Yts[2])
    mul!(Yts[2],s.Phi,s.R'); msub!(Yts[1],Yts[1],Yts[2])
    mul!(Ms[2], s.X_tilde', s.X_tilde); Ms[2][diagind(Ms[2])] .+= phi
    mul!(s.Y_tilde,inv(Ms[2]),Yts[1])

    # U .= s.L*s.X_tilde+s.Lambda/lambda
    mul!(Us[1],s.L,s.X_tilde)
    sdiv!(Us[2],s.Lambda,lambda)
    madd!(U,Us[1],Us[2])
    # V .= s.Y_tilde*s.R+s.Phi/phi
    mul!(Vs[1],s.Y_tilde,s.R)
    sdiv!(Vs[2],s.Phi,phi)
    madd!(V,Vs[1],Vs[2])

    mnonneg!(U)
    mnonneg!(V)

    # s.Lambda .+= xi*lambda*(s.L*s.X_tilde-U)
    msub!(Us[1],Us[1],U)
    rmul!(Us[1],xilambda); madd!(s.Lambda,s.Lambda,Us[1])
    # s.Phi .+= xi*phi*(s.Y_tilde*s.R-V)
    msub!(Vs[1],Vs[1],V)
    rmul!(Vs[1],xiphi); madd!(s.Phi,s.Phi,Vs[1])
end

function update_wh_slow!(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, V) where T
    r = size(U,2)
    xi = updater.xi; lambda = updater.lambda; phi = updater.phi

    s.X_tilde .= (s.A_tilde*s.Y_tilde'+lambda*s.L'U-s.L's.Lambda)*inv(s.Y_tilde*s.Y_tilde'+lambda*Matrix(1.0I,r,r))
    s.Y_tilde .= inv(s.X_tilde's.X_tilde+phi*Matrix(1.0I,r,r))*(s.X_tilde's.A_tilde+phi*V*s.R'-s.Phi*s.R')

    U .= s.L*s.X_tilde+s.Lambda/lambda
    V .= s.Y_tilde*s.R+s.Phi/phi

    mnonneg!(U)
    mnonneg!(V)

    s.Lambda .+= xi*lambda*(s.L*s.X_tilde-U)
    s.Phi .+= xi*phi*(s.Y_tilde*s.R-V)
end

function update_wh_cnmf!(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, V; X=nothing, Y=nothing, ls=0) where T
    L = s.L
    R = s.R
    A = s.A_tilde
    r = size(U,2)
    m = size(L,1)
    n = size(R,2)
    Y = V*R'
    Lam = zeros(size(U))
    Phi = zeros(size(V))
    l = 1.
    f = 1.
    x = 1.
    Idnty = Matrix(1.0I,r,r)
    iter = 0
    while iter < 1000
        iter += 1
        X = ((Y*Y' + l*Idnty)\(Y*A' + (l*U' - Lam')*L))'
        # Y_tilde .= inv(X_tilde'X_tilde+phi*I)*(X_tilde'A_tilde+phi*V*R'-Phi*R')
        Y = (X'X + f*Idnty)\(X'A + (f*V - Phi .- ls)*R')
        # Utmp = L*X_tilde+Lambda/lambda
        LX = L*X
        U .= LX + Lam/l
        U[U.<0] .= 0
        # Vtmp = Y_tilde*R+Phi/phi
        YR = Y*R
        V .= YR + Phi/f
        V[V.<0] .= 0
        # Lambda .+= xi*lambda*(L*X_tilde-U)
        # Phi .+= xi*phi*(Y_tilde*R-V)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - V)
    end
    s.X_tilde .= X; s.Y_tilde .= Y
end

function update_wh0!(A, L, R, U, V, r; X=nothing, Y=nothing, max_iter=100, ls=0)
    Y = V*R'
    Lam = zeros(size(U))
    Phi = zeros(size(V))
    l = 1.
    f = 1.
    x = 1.
    Idnty = Matrix(1.0I,r,r)
    it = 0
    while it < max_iter
        it += 1
        # X_tilde .= (A_tilde*Y_tilde'+lambda*L'U-L'Lambda)*inv(Y_tilde*Y_tilde'+lambda*I)
        X = ((Y*Y' + l*Idnty)\(Y*A' + (l*U' - Lam')*L))'
        # Y_tilde .= inv(X_tilde'X_tilde+phi*I)*(X_tilde'A_tilde+phi*V*R'-Phi*R')
        Y = (X'X + f*Idnty)\(X'A + (f*V - Phi .- ls)*R')
        # Utmp = L*X_tilde+Lambda/lambda
        LX = L*X
        U = LX + Lam/l
        U[U.<0] .= 0
        # Vtmp = Y_tilde*R+Phi/phi
        YR = Y*R
        V = YR + Phi/f
        V[V.<0] .= 0
        # Lambda .+= xi*lambda*(L*X_tilde-U)
        # Phi .+= xi*phi*(Y_tilde*R-V)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - V)
    end
    return X, Y
end

"""Implements compressed NMF using an ADMM method as described in
Tepper and Shapiro, IEEE TSP 2015
min_{U,V,X,Y} ||A - XY||_F^2 s.t. U = LX >= 0 and V = YR >=0
"""
function compressive_nmf_cnmf(A, L, R, r; X=nothing, Y=nothing, max_iter=100, ls=0)
    m = size(L,1)
    n = size(R,2)
    U = rand(m, r)
    V = rand(r, n)
    Y = V*R'
    Lam = zeros(size(U))
    Phi = zeros(size(V))
    l = 1.
    f = 1.
    x = 1.
    Idnty = Matrix(1.0I,r,r)
    it = 0
    while it < max_iter
        it += 1
        # X_tilde .= (A_tilde*Y_tilde'+lambda*L'U-L'Lambda)*inv(Y_tilde*Y_tilde'+lambda*I)
        X = ((Y*Y' + l*Idnty)\(Y*A' + (l*U' - Lam')*L))'
        # Y_tilde .= inv(X_tilde'X_tilde+phi*I)*(X_tilde'A_tilde+phi*V*R'-Phi*R')
        Y = (X'X + f*Idnty)\(X'A + (f*V - Phi .- ls)*R')
        # Utmp = L*X_tilde+Lambda/lambda
        LX = L*X
        U = LX + Lam/l
        U[U.<0] .= 0
        # Vtmp = Y_tilde*R+Phi/phi
        YR = Y*R
        V = YR + Phi/f
        V[V.<0] .= 0
        # Lambda .+= xi*lambda*(L*X_tilde-U)
        # Phi .+= xi*phi*(Y_tilde*R-V)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - V)
    end
    return X, Y
end

function compressive_nmf(A, L, R, U, V, r; X=nothing, Y=nothing, max_iter=100, ls=0)
    Y = V*R'
    Lam = zeros(size(U))
    Phi = zeros(size(V))
    l = 1.
    f = 1.
    x = 1.
    Idnty = Matrix(1.0I,r,r)
    it = 0
    while it < max_iter
        it += 1
        # X_tilde .= (A_tilde*Y_tilde'+lambda*L'U-L'Lambda)*inv(Y_tilde*Y_tilde'+lambda*I)
        X = ((Y*Y' + l*Idnty)\(Y*A' + (l*U' - Lam')*L))'
        # Y_tilde .= inv(X_tilde'X_tilde+phi*I)*(X_tilde'A_tilde+phi*V*R'-Phi*R')
        Y = (X'X + f*Idnty)\(X'A + (f*V - Phi .- ls)*R')
        # Utmp = L*X_tilde+Lambda/lambda
        LX = L*X
        U = LX + Lam/l
        U[U.<0] .= 0
        # Vtmp = Y_tilde*R+Phi/phi
        YR = Y*R
        V = YR + Phi/f
        V[V.<0] .= 0
        # Lambda .+= xi*lambda*(L*X_tilde-U)
        # Phi .+= xi*phi*(Y_tilde*R-V)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - V)
    end
    return X, Y
end

end
