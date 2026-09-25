module CompNMF

using LinearAlgebra, NMF, RandomizedLinAlg, DataStructures, StatsBase
using AverageFits
export solve!, CompressedNMF, compmat

mutable struct CompressedNMF{T}
    maxiter::Int           # maximum number of iterations (in main procedure)
    verbose::Bool          # whether to show procedural information
    tol::T                 # tolerance of changes on U and Vt upon convergence
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
    normU::T
    normVt::T
    X_tilde::Matrix{T}
    Y_tilde::Matrix{T}
    Lambda::Matrix{T}
    Phi::Matrix{T}
    TmpUs::Vector{Matrix{T}}
    TmpVts::Vector{Matrix{T}}
    TmpXts::Vector{SubArray{T}}
    TmpYts::Vector{SubArray{T}}
    TmpMs::Vector{SubArray{T}}
    function CompressedNMFState{T}(X_tilde, Y_tilde, L, R, A_tilde) where T
        m, rrov = size(L); rrov, n = size(R); r = size(X_tilde,2)
        Lambda, Phi = zeros(T,m,r), zeros(T,r,n)
        TmpUs = map(i->Matrix{T}(undef,m,r),1:2)
        TmpVts = map(i->Matrix{T}(undef,r,n),1:2)
        Ms = map(i->Matrix{T}(undef,rrov,rrov),1:2)
        TmpXts = map(i->view(Ms[i],1:rrov,1:r),1:2)
        TmpYts = map(i->view(Ms[i],1:r,1:rrov),1:2)
        TmpMs = map(i->view(Ms[i],1:r,1:r),1:2)
        new{T}(L, R, A_tilde, 0., 0., X_tilde, Y_tilde,
               Lambda, Phi, TmpUs, TmpVts, TmpXts, TmpYts, TmpMs)
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
    wavgfits::Vector{T}
    corrs::Vector{T}    # left-factor (U) matched ground-truth correlation (AverageFits.wh_correlations); == wcorrs
    wcorrs::Vector{T}   # left  (U)  factor matched ground-truth correlation
    hcorrs::Vector{T}   # right (Vt) factor matched ground-truth correlation, on the SAME matching as the left
    inittime::T
    function Result{T}(L::Matrix{T}, R::Matrix{T}, A_tilde::Matrix{T}, X_tilde::Matrix{T}, Y_tilde::Matrix{T},
            niters::Int, converged::Bool, objv, objvs, sparsevalues, avgfits, wavgfits, corrs, wcorrs, hcorrs, inittime) where T
       new{T}(L, R, A_tilde, X_tilde, Y_tilde, niters, converged, objv, objvs, sparsevalues, avgfits, wavgfits, corrs, wcorrs, hcorrs, inittime)
    end
end

function prepare_state(::CompressedNMFUpd{T}, A, U, Vt; L=nothing, R=nothing) where T
    time0 = time()
    if (L === nothing) || (R === nothing)
        L, R, X_tilde, Y_tilde, A_tilde = compmat(A, U, Vt; w=4)
    else
        A_tilde = L'*A*R'
        X_tilde, Y_tilde = L'U, Vt*R'
    end
    inittime = time()-time0
    state = CompressedNMFState{T}(X_tilde, Y_tilde, L, R, A_tilde)
    state, inittime # U and Vt are not stored
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

function compmat(A::AbstractArray{T,2}, Up, Vtp; w=4, rov=10) where T
    r = size(Up,2)
    L = low_rank_QR(A,r+rov,w=w)
    R = Array(low_rank_QR(A',r+rov,w=w)')
    A_tilde = L'*A*R'
    # balanceUVt!(Up,Vtp)
    X_tilde, Y_tilde = L'Up, Vtp*R'
    # @show norm(A-L*L'A*R'R)^2
    L, R, X_tilde, Y_tilde, A_tilde
end

function balanceUVt!(Un, Vtn)
    for k in 1:size(Un,2)
        balanceUkVtk!(view(Un,:,k), view(Vtn,k,:))
    end
    Un, Vtn
end
function balanceUkVtk!(Uk, Vtk)
    normw = max(eps(eltype(Uk)),norm(Uk))
    normh = max(eps(eltype(Vtk)),norm(Vtk))
    balfacs = sqrt(normw/normh)
    Uk ./= balfacs; Vtk .*= balfacs
end

solve!(alg::CompressedNMF{T}, A, U, Vt; L=nothing,R=nothing,
        gtU::Matrix{T}=Matrix{T}(undef,0,0), gtVt::Matrix{T}=Matrix{T}(undef,0,0),
        maskU::Union{Colon,Vector,BitVector}=Colon(),maskVt::Union{Colon,Vector,BitVector}=Colon(),
        corr_primary::Symbol=:WH, sub_base_q=0.01, delta_f=false, weighted=true) where {T} =
    nmf_skeleton!(CompressedNMFUpd{T}(alg.xi, alg.lambda, alg.phi, alg.SCA_penmetric, alg.SCA_αw, alg.SCA_αh),
            A, U, Vt, alg.maxiter, alg.verbose, alg.tol; L=L, R=R, gtU=gtU, gtVt=gtVt, maskU=maskU, maskVt=maskVt,
            corr_primary=corr_primary, sub_base_q=sub_base_q, delta_f=delta_f, weighted=weighted)

function evaluate_objv(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, Vt) where T
    # convert(T, 0.5) * sqL2dist(A, s.UVt)
    if updater.SCA_penmetric ∈ [:HALS, :SPARSE_U, :SPARSE_Vt]
        norm(A-U*Vt)^2
    elseif updater.SCA_penmetric == :SCA
        X_tilde = s.L'*U; Y_tilde = Vt*s.R'
        norm(Diagonal(s.A_tilde)-X_tilde*Y_tilde)^2
    elseif updater.SCA_penmetric == :CompNMF
        p1 = norm(s.A_tilde-s.X_tilde*s.Y_tilde)^2
        p2 = sum(s.Lambda.*(s.L*s.X_tilde-U))
        p3 = updater.lambda/2*norm(s.L*s.X_tilde-U)^2
        p4 = sum(s.Phi.*(s.Y_tilde*s.R-Vt))
        p5 = updater.phi/2*norm(s.Y_tilde*s.R-Vt)^2
        p1 + p2 + p3 + p4 + p5
    end
end
function evaluate_sparseness(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, Vt) where T
    if updater.SCA_penmetric == :SCA
        X_tilde = s.L'*U; Y_tilde = Vt*s.R'
        normL1 = norm(s.L,1); normR1 = norm(s.R,1); (αw, αh) = (updater.SCA_αw/normL1, updater.SCA_αh/normR1)
        αw*norm(s.L*X_tilde,1) + αh*norm(Y_tilde*s.R,1)
    elseif updater.SCA_penmetric == :SPARSE_U
        Un, Vtn = copy(U), copy(Vt); normalizeU!(Un,Vtn)
        norm(Un,1)#/s.normU
    elseif updater.SCA_penmetric == :SPARSE_Vt
        norm(Vt,1)#/s.normVt
    else
        zero(T)
    end
end

function nmf_skeleton!(updater::NMF.NMFUpdater{T},
                       A, U::Matrix{T}, Vt::Matrix{T},
                       maxiter::Int, verbose::Bool, tol;
                       L=nothing, R=nothing,
                       gtU::Matrix{T}=Matrix{T}(undef,0,0),
                       gtVt::Matrix{T}=Matrix{T}(undef,0,0),
                       maskU::Union{Colon,Vector,BitVector}=Colon(),
                       maskVt::Union{Colon,Vector,BitVector}=Colon(),
                       corr_primary::Symbol=:WH,
                       sub_base_q=0.01, delta_f=false, weighted=true
                       ) where T
    objv = convert(T, NaN)
    # init

    state, inittime = prepare_state(updater, A, U, Vt; L=L, R=R)
    preU = Matrix{T}(undef, size(U))
    preVt = Matrix{T}(undef, size(Vt))
    objvs = T[]; objvsparses = T[]; avgfits=T[]; wavgfits=T[]; corrs=T[]; wcorrs=T[]; hcorrs=T[]
    gtVtt = permutedims(gtVt)   # right-factor GT (n x K), computed once for the loop
    # Both-factor matched ground-truth correlation on ONE shared matching, via
    # AverageFits.wh_correlations: `corr_primary` (:W / :H) picks which factor is matched
    # first (U = left / Vt = right); the other reuses that assignment (Vt' ==
    # permutedims(Vt)) so both R values describe the same gt<->component pairing.
    # CompNMF is non-negative → no sign-flip.
    wh_corr(U, Vt) = wh_correlations(gtU, gtVtt, U, permutedims(Vt); maskW = maskU, maskH = maskVt, primary = corr_primary)
    if verbose
        start = time()
        objv = evaluate_objv(updater, state, A, U, Vt)
        push!(objvs,objv)
        push!(objvsparses,evaluate_sparseness(updater, state, A, U, Vt))
        push!(avgfits, evaluate_fitvalue(gtU, gtVt, A, U, Vt, maskU, maskVt; sub_base_q=sub_base_q, delta_f=delta_f, weighted=false)[1])
        push!(wavgfits, evaluate_fitvalue(gtU, gtVt, A, U, Vt, maskU, maskVt; sub_base_q=sub_base_q, delta_f=delta_f, weighted=true)[1])
        wc, hc = wh_corr(U, Vt); push!(corrs, wc); push!(wcorrs, wc); push!(hcorrs, hc)
        # @printf("%-5s    %-13s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change", "(U & Vt).change")
        # @printf("%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
    end

    # main loop
    converged = false
    iter = 0
    while !converged && iter < maxiter
        iter += 1
        copyto!(preU, U)
        copyto!(preVt, Vt)

        # update Vt
        update_wh!(updater, state, A, U, Vt)

        # determine convergence
        dev = max(maxad(preU, U), maxad(preVt, Vt))
        if dev < tol
            converged = true
        end

        # display info
        if verbose
            elapsed = time() - start
            preobjv = objv
            objv = evaluate_objv(updater, state, A, U, Vt)
            push!(objvs,objv)
            push!(objvsparses,evaluate_sparseness(updater, state, A, U, Vt))
            push!(avgfits, evaluate_fitvalue(gtU, gtVt, A, U, Vt, maskU, maskVt; sub_base_q=sub_base_q, delta_f=delta_f, weighted=false)[1])
            push!(wavgfits, evaluate_fitvalue(gtU, gtVt, A, U, Vt, maskU, maskVt; sub_base_q=sub_base_q, delta_f=delta_f, weighted=true)[1])
            wc, hc = wh_corr(U, Vt); push!(corrs, wc); push!(wcorrs, wc); push!(hcorrs, hc)
            #@printf("%5d    %13.6e    %13.6e    %13.6e    %13.6e\n",
            #    t, elapsed, objv, objv - preobjv, dev)
        end
    end
    if !verbose
        objv = evaluate_objv(updater, state, A, U, Vt)
    end
 #   return Result{T}(U, Vt, iter, converged, objv, objvs, objvsparses, avgfits, inittime)
    return Result{T}(state.L, state.R, state.A_tilde, state.X_tilde, state.Y_tilde, iter,
                    converged, objv, objvs, objvsparses, avgfits, wavgfits, corrs, wcorrs, hcorrs, inittime)
end

function update_wh!(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, Vt) where T
    Us = s.TmpUs; Vts = s.TmpVts; Xts = s.TmpXts; Yts = s.TmpYts; Ms = s.TmpMs
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

    # s.Y_tilde .= inv(s.X_tilde's.X_tilde+phi*Matrix(1.0I,r,r))*(s.X_tilde's.A_tilde+phi*Vt*s.R'-s.Phi*s.R')
    mul!(Yts[1],s.X_tilde',s.A_tilde)
    mul!(Yts[2],Vt,s.R'); rmul!(Yts[2],phi)
    madd!(Yts[1],Yts[1],Yts[2])
    mul!(Yts[2],s.Phi,s.R'); msub!(Yts[1],Yts[1],Yts[2])
    mul!(Ms[2], s.X_tilde', s.X_tilde); Ms[2][diagind(Ms[2])] .+= phi
    mul!(s.Y_tilde,inv(Ms[2]),Yts[1])

    # U .= s.L*s.X_tilde+s.Lambda/lambda
    mul!(Us[1],s.L,s.X_tilde)
    sdiv!(Us[2],s.Lambda,lambda)
    madd!(U,Us[1],Us[2])
    # Vt .= s.Y_tilde*s.R+s.Phi/phi
    mul!(Vts[1],s.Y_tilde,s.R)
    sdiv!(Vts[2],s.Phi,phi)
    madd!(Vt,Vts[1],Vts[2])

    mnonneg!(U)
    mnonneg!(Vt)

    # s.Lambda .+= xi*lambda*(s.L*s.X_tilde-U)
    msub!(Us[1],Us[1],U)
    rmul!(Us[1],xilambda); madd!(s.Lambda,s.Lambda,Us[1])
    # s.Phi .+= xi*phi*(s.Y_tilde*s.R-Vt)
    msub!(Vts[1],Vts[1],Vt)
    rmul!(Vts[1],xiphi); madd!(s.Phi,s.Phi,Vts[1])
end

function update_wh_slow!(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, Vt) where T
    r = size(U,2)
    xi = updater.xi; lambda = updater.lambda; phi = updater.phi

    s.X_tilde .= (s.A_tilde*s.Y_tilde'+lambda*s.L'U-s.L's.Lambda)*inv(s.Y_tilde*s.Y_tilde'+lambda*Matrix(1.0I,r,r))
    s.Y_tilde .= inv(s.X_tilde's.X_tilde+phi*Matrix(1.0I,r,r))*(s.X_tilde's.A_tilde+phi*Vt*s.R'-s.Phi*s.R')

    U .= s.L*s.X_tilde+s.Lambda/lambda
    Vt .= s.Y_tilde*s.R+s.Phi/phi

    mnonneg!(U)
    mnonneg!(Vt)

    s.Lambda .+= xi*lambda*(s.L*s.X_tilde-U)
    s.Phi .+= xi*phi*(s.Y_tilde*s.R-Vt)
end

function update_wh_cnmf!(updater::CompressedNMFUpd{T}, s::CompressedNMFState{T}, A, U, Vt; X=nothing, Y=nothing, ls=0) where T
    L = s.L
    R = s.R
    A = s.A_tilde
    r = size(U,2)
    m = size(L,1)
    n = size(R,2)
    Y = Vt*R'
    Lam = zeros(size(U))
    Phi = zeros(size(Vt))
    l = 1.
    f = 1.
    x = 1.
    Idnty = Matrix(1.0I,r,r)
    iter = 0
    while iter < 1000
        iter += 1
        X = ((Y*Y' + l*Idnty)\(Y*A' + (l*U' - Lam')*L))'
        # Y_tilde .= inv(X_tilde'X_tilde+phi*I)*(X_tilde'A_tilde+phi*Vt*R'-Phi*R')
        Y = (X'X + f*Idnty)\(X'A + (f*Vt - Phi .- ls)*R')
        # Utmp = L*X_tilde+Lambda/lambda
        LX = L*X
        U .= LX + Lam/l
        U[U.<0] .= 0
        # Vttmp = Y_tilde*R+Phi/phi
        YR = Y*R
        Vt .= YR + Phi/f
        Vt[Vt.<0] .= 0
        # Lambda .+= xi*lambda*(L*X_tilde-U)
        # Phi .+= xi*phi*(Y_tilde*R-Vt)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - Vt)
    end
    s.X_tilde .= X; s.Y_tilde .= Y
end

function update_wh0!(A, L, R, U, Vt, r; X=nothing, Y=nothing, max_iter=100, ls=0)
    Y = Vt*R'
    Lam = zeros(size(U))
    Phi = zeros(size(Vt))
    l = 1.
    f = 1.
    x = 1.
    Idnty = Matrix(1.0I,r,r)
    it = 0
    while it < max_iter
        it += 1
        # X_tilde .= (A_tilde*Y_tilde'+lambda*L'U-L'Lambda)*inv(Y_tilde*Y_tilde'+lambda*I)
        X = ((Y*Y' + l*Idnty)\(Y*A' + (l*U' - Lam')*L))'
        # Y_tilde .= inv(X_tilde'X_tilde+phi*I)*(X_tilde'A_tilde+phi*Vt*R'-Phi*R')
        Y = (X'X + f*Idnty)\(X'A + (f*Vt - Phi .- ls)*R')
        # Utmp = L*X_tilde+Lambda/lambda
        LX = L*X
        U = LX + Lam/l
        U[U.<0] .= 0
        # Vttmp = Y_tilde*R+Phi/phi
        YR = Y*R
        Vt = YR + Phi/f
        Vt[Vt.<0] .= 0
        # Lambda .+= xi*lambda*(L*X_tilde-U)
        # Phi .+= xi*phi*(Y_tilde*R-Vt)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - Vt)
    end
    return X, Y
end

"""Implements compressed NMF using an ADMM method as described in
Tepper and Shapiro, IEEE TSP 2015
min_{U,Vt,X,Y} ||A - XY||_F^2 s.t. U = LX >= 0 and Vt = YR >=0
"""
function compressive_nmf_cnmf(A, L, R, r; X=nothing, Y=nothing, max_iter=100, ls=0)
    m = size(L,1)
    n = size(R,2)
    U = rand(m, r)
    Vt = rand(r, n)
    Y = Vt*R'
    Lam = zeros(size(U))
    Phi = zeros(size(Vt))
    l = 1.
    f = 1.
    x = 1.
    Idnty = Matrix(1.0I,r,r)
    it = 0
    while it < max_iter
        it += 1
        # X_tilde .= (A_tilde*Y_tilde'+lambda*L'U-L'Lambda)*inv(Y_tilde*Y_tilde'+lambda*I)
        X = ((Y*Y' + l*Idnty)\(Y*A' + (l*U' - Lam')*L))'
        # Y_tilde .= inv(X_tilde'X_tilde+phi*I)*(X_tilde'A_tilde+phi*Vt*R'-Phi*R')
        Y = (X'X + f*Idnty)\(X'A + (f*Vt - Phi .- ls)*R')
        # Utmp = L*X_tilde+Lambda/lambda
        LX = L*X
        U = LX + Lam/l
        U[U.<0] .= 0
        # Vttmp = Y_tilde*R+Phi/phi
        YR = Y*R
        Vt = YR + Phi/f
        Vt[Vt.<0] .= 0
        # Lambda .+= xi*lambda*(L*X_tilde-U)
        # Phi .+= xi*phi*(Y_tilde*R-Vt)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - Vt)
    end
    return X, Y
end

function compressive_nmf(A, L, R, U, Vt, r; X=nothing, Y=nothing, max_iter=100, ls=0)
    Y = Vt*R'
    Lam = zeros(size(U))
    Phi = zeros(size(Vt))
    l = 1.
    f = 1.
    x = 1.
    Idnty = Matrix(1.0I,r,r)
    it = 0
    while it < max_iter
        it += 1
        # X_tilde .= (A_tilde*Y_tilde'+lambda*L'U-L'Lambda)*inv(Y_tilde*Y_tilde'+lambda*I)
        X = ((Y*Y' + l*Idnty)\(Y*A' + (l*U' - Lam')*L))'
        # Y_tilde .= inv(X_tilde'X_tilde+phi*I)*(X_tilde'A_tilde+phi*Vt*R'-Phi*R')
        Y = (X'X + f*Idnty)\(X'A + (f*Vt - Phi .- ls)*R')
        # Utmp = L*X_tilde+Lambda/lambda
        LX = L*X
        U = LX + Lam/l
        U[U.<0] .= 0
        # Vttmp = Y_tilde*R+Phi/phi
        YR = Y*R
        Vt = YR + Phi/f
        Vt[Vt.<0] .= 0
        # Lambda .+= xi*lambda*(L*X_tilde-U)
        # Phi .+= xi*phi*(Y_tilde*R-Vt)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - Vt)
    end
    return X, Y
end

end
