function normalizeU!(U,V)
    p = size(U,2)
    for i = 1:p
        nrm = max(eps(eltype(U)),norm(U[:,i]))
        if nrm != 0.
            U[:,i] ./= nrm
            V[i,:] .*= nrm
        end
    end
    U,V
end

fitx(a,b) = (m=sum(a)/length(a); denom=sum(abs2,a.-m); fitx(a,b,denom))
fitx(a,b,denom) = (1-sum(abs2,a-b)/denom)
fitd(a,b) = (na=norm(a); nb=norm(b); fitd(a,b,na,nb))
fitd(a,b,na) = (nb=norm(b); fitd(a,b,na,nb))
fitd(a,b,na,nb) = (denom=na^2+nb^2+2na*nb; (1-sum(abs2,a-b)/denom,denom))
#fitd(a,b,na,nb) = (denom=na^2+nb^2+2na*nb; 1-sum(abs2,a-b)/denom)
fitd(a,b,nbn,na,nb) = (denom=na^2+nb^2+2na*nb; (1-(sum(abs2,a-b)+nbn^2)/denom,denom))
# calfit(a,b) = (dval=fitd(a,b); aval=fitd(a,-b); dval > aval ? (dval, false) : (aval, true))
# calfit(a,b,na) = (dval=fitd(a,b,na); aval=fitd(a,-b,na); dval > aval ? (dval, false) : (aval, true))
ssd(a,b) = sum(abs2,a-b)
nssd(a,b) = (ssd(a,b)/(norm(a)*norm(b)), false)
nssda(a,b) = (ssdval=ssd(a,b); ssaval=ssd(a,-b); nab=(norm(a)*norm(b));
                ssdval < ssaval ? (ssdval/nab, false) : (ssaval/nab, true))
function fiterr(a,b)
    init_x = eltype(a)[1, 0]
    f(x) = norm((x[1].*b.+x[2]).-a)^2
    rst = optimize(f,init_x)
    rst.minimum/length(a), false
end

"""
    matchlist, ssds = matchcomponents(GT, W, errorfn::Function)
Matched the W columns with with those of GT.
GT: ground truch matrix
W: matrix to Compare
errorfn: function used to calculate the error of two vectors
matchlist: list of pair = (column index of GT, column index of W)
ssds: list of ssd for the matched pair columns
"""
function matchWcomponents(GT, W, errorfn::Function) # M X r form
    pq = PriorityQueue{Tuple{Int,Int,Bool}, Float64}(Base.Order.Forward) # Forward(low->high)
    gtcolnum = size(GT,2); wcolnum = size(W,2)
    for i = 1:gtcolnum
        gti = GT[:,i]
        for j = 1:wcolnum
            wj = W[:,j]
            dist, invert = errorfn(gti,wj)
            enqueue!(pq,(i,j,invert),dist)
        end
    end
    matchlist = Tuple{Int,Int,Bool}[]
    errs = Float64[]
    while !isempty(pq)
        p = peek(pq)
        dequeue!(pq)
        found = false
        mllength = length(matchlist)
        for i = 1:mllength
            if p[1][1] == matchlist[i][1] || p[1][2] == matchlist[i][2]
                found = true
                break
            end
        end
        if !found
            push!(matchlist,(p[1][1],p[1][2],p[1][3]))
            push!(errs,p[2])
        end
    end
    matchlist, errs
end


function matchedorder(GTW::AbstractArray{T}, GTH::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T},
            noc; clamp=false, iscalunmatched=false, sdsr=1, tdsr=1) where T
    pq = PriorityQueue{Tuple{Int,Int,Bool,T}, T}(Base.Order.Reverse) # Reverse(high->low)
    gtcolnum = size(GTW,2); wcolnum = size(W,2)
    gtWsum = dropdims(sum(abs,GTW, dims=2), dims=2); gtHsum = dropdims(sum(abs,GTH, dims=1), dims=1)
    iw = gtWsum.!=0; ih = gtHsum.!=0 # to reduce computation, choose only non-zero rows and column

    for i = 1:gtcolnum
        gtwi = GTW[iw,i]; gthi = GTH[i,ih]; gtxi = gtwi[1:sdsr:end]*gthi[1:tdsr:end]'; ngtxi = norm(gtxi)
        for j = 1:wcolnum
            wj = W[iw,j]; hj = H[j,ih]; xj = wj[1:sdsr:end]*hj[1:tdsr:end]'
            clamp && (xj[xj.<0].=0)
            fitval, denom = fitd(gtxi,xj,ngtxi)
            enqueue!(pq,(i,j,false,denom),fitval)
        end
    end
    matchlist = Tuple{Int,Int,Bool}[]; ml = Int[]
    fitvals = T[]; denomsum = 0.
    while !isempty(pq)
        p = peek(pq)
        dequeue!(pq)
        found = false
        mllength = length(matchlist)
        for i = 1:mllength
            if p[1][1] == matchlist[i][1] || p[1][2] == matchlist[i][2]
                found = true
                break
            end
        end
        if !found
            push!(matchlist,(p[1][1],p[1][2],p[1][3]))
            push!(ml,p[1][2])
            push!(fitvals,p[2]*p[1][4]) # power weighted fitval
            denomsum += p[1][4]
        end
    end
    # Calculate unmatched power
    rerrs = T[]
    if iscalunmatched
        unmatchlist = collect(1:wcolnum)
        filter!(a->a ∉ ml,unmatchlist)
        gtxi = zeros(T,size(W,1),size(H,2))
        for j in unmatchlist
            wj = W[iw...,j]; hj = H[j,ih...]; xj = wj*hj'
            rerr = sum(abs2,xj)
            push!(rerrs,rerr)
        end
    end
    nodr = matchedorder(matchlist, noc)
    nodr, matchlist, fitvals, rerrs, denomsum
end

function fitcomponents(GTW::AbstractArray{T}, GTH::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T};
            clamp=false, iscalunmatched=false, sdsr=1, tdsr=1) where T
    pq = PriorityQueue{Tuple{Int,Int,Bool,T}, T}(Base.Order.Reverse) # Reverse(high->low)
    gtcolnum = size(GTW,2); wcolnum = size(W,2)
    for i = 1:gtcolnum
        gtwi = GTW[:,i]; gthi = GTH[i,:]; gtxi = gtwi[1:sdsr:end]*gthi[1:tdsr:end]'; ngtxi = norm(gtxi)
        for j = 1:wcolnum
            wj = W[:,j]; hj = H[j,:]; xj = wj[1:sdsr:end]*hj[1:tdsr:end]'
            clamp && (xj[xj.<0].=0)
            fitval, denom = fitd(gtxi,xj,ngtxi)
            enqueue!(pq,(i,j,false,denom),fitval)
        end
    end
    matchlist = Tuple{Int,Int,Bool}[]; ml = Int[]
    fitvals = T[]; denomsum = 0.
    while !isempty(pq)
        p = peek(pq)
        dequeue!(pq)
        found = false
        mllength = length(matchlist)
        for i = 1:mllength
            if p[1][1] == matchlist[i][1] || p[1][2] == matchlist[i][2]
                found = true
                break
            end
        end
        if !found
            push!(matchlist,(p[1][1],p[1][2],p[1][3]))
            push!(ml,p[1][2])
            push!(fitvals,p[2]*p[1][4]) # power weighted fitval
            denomsum += p[1][4]
        end
    end
    # Calculate unmatched power
    rerrs = T[]
    if iscalunmatched
        unmatchlist = collect(1:wcolnum)
        filter!(a->a ∉ ml,unmatchlist)
        gtxi = zeros(T,size(W,1),size(H,2))
        for j in unmatchlist
            wj = W[:,j]; hj = H[j,:]; xj = wj*hj'
            rerr = sum(abs2,xj)
            push!(rerrs,rerr)
        end
    end
    matchlist, fitvals, rerrs, denomsum
end

function fitcomponents(X::AbstractArray, GTX::AbstractVector, W::AbstractArray{T}, H::AbstractArray{T};
            clamp=false, ordered=false) where T
    pq = PriorityQueue{Tuple{Int,Int,Bool,T}, T}(Base.Order.Reverse) # Reverse(high->low)
    gtcolnum = length(GTX); wcolnum = size(W,2); allidxs = collect(1:size(W,1))
    fitvals = T[]; denomsum = 0.
    for i = 1:gtcolnum
        # Read bounding box from GTX[i][2] and mask from GTX[i][3]
        # Then, perform masking with mask in the bounding box
        # Ground truth X values of all the outside of the mask are assumed to be zero.
        vecs = vcat(collect.(GTX[i][2])...); idxs = vecs[Bool.(GTX[i][3])] # TODO: make eltype(GTX[i][3]) as Bool
        gtxi = X[idxs,:]; ngtxi = norm(gtxi)
        # calcuate fit value  with each W[:,j]*H[j,:]'
        if ordered
            wi = W[idxs,i]; hi = H[i,:]; xi = wi*hi'
            nxi2 = norm(xi)^2; nxiall2 = norm(W[:,i]*H[i,:]')^2; nxin2 = nxiall2-nxi2
            clamp && (xi[xi.<0].=0)
            fitval, denom = fitd(gtxi,xi,sqrt(nxin2),sqrt(nxiall2),ngtxi) # fitval = fitd(gtxi,xj,nxjn,ngtxi)
            push!(fitvals,fitval*denom) # power weighted fitval
            denomsum += denom
        else
            for j = 1:wcolnum
                wj = W[idxs,j]; hj = H[j,:]; xj = wj*hj' # when mask value is 1
                # Calculate the norm of wjn*hj' which is the X of mask vlaue is 0
                # wjn = W[allidxs[allidxs.∉ [idxs]],j] # when mask value is 0
                # s=0; for w = wjn, h = hj s += (w*h)^2 end; nxjn = sqrt(s)
                nxj2 = norm(xj)^2; nxjall2 = norm(W[:,j]*H[j,:]')^2; nxjn2 = nxjall2-nxj2
                clamp && (xj[xj.<0].=0)
                fitval, denom = fitd(gtxi,xj,sqrt(nxjn2),sqrt(nxjall2),ngtxi) # fitval = fitd(gtxi,xj,nxjn,ngtxi)

                # norm2diff = norm(gtxi-xj)^2; norm2outside = nxjn2
                # norm2nom = norm2diff+norm2outside
                # norm2gt = ngtxi^2; norm2xj = norm(xj)^2; norm2xjall = norm2xj+norm2outside
                # twonorm2gtxjall = 2*ngtxi*sqrt(norm2xjall); norm2denom = norm2gt+norm2xjall+twonorm2gtxjall
                # @show norm2diff, norm2outside, norm2nom
                # @show norm2gt, norm2xj, norm2xjall, twonorm2gtxjall, norm2denom
                # @show fitval, norm2nom/norm2denom, 1-norm2nom/norm2denom

                enqueue!(pq,(i,j,false,denom),fitval)
            end
        end
    end
    matchlist = Tuple{Int,Int,Bool}[]#; ml = Int[]

    if !ordered
        # find best matched pair (i,j)
        while !isempty(pq)
            p = peek(pq)
            dequeue!(pq)
            found = false
            mllength = length(matchlist)
            for i = 1:mllength
                if p[1][1] == matchlist[i][1] || p[1][2] == matchlist[i][2]
                    found = true
                    break
                end
            end
            if !found
                push!(matchlist,(p[1][1],p[1][2],p[1][3]))
                # push!(ml,p[1][2])
                push!(fitvals,p[2]*p[1][4]) # power weighted fitval
                denomsum += p[1][4]
            end
        end
    else
        # foreach(i->push!(matchlist,(i,i,false)), 1:gtcolnum)
    end
    rerrs = T[]
    # gtindices = map(i->matchlist[i][1],1:gtcolnum)
    # @show fitvals[sortperm(gtindices)], sortperm(gtindices)
    matchlist, fitvals, rerrs, denomsum
end

matchedWnssd(GT,W) = ((ml, nssds) = matchWcomponents(GT, W, nssd); (sum(nssds)/length(nssds), ml, nssds))
matchedWnssda(GT,W) = ((ml, nssdas) = matchWcomponents(GT, W, nssda); (sum(nssdas)/length(nssdas), ml, nssdas))
matchedfitval(GTW, GTH, W, H; clamp=false, maskW=Colon(), maskH=Colon(), sdsr=1, tdsr=1) =
    ((ml, fitvals, rerrs, denomsum) = fitcomponents(GTW, GTH, W, H; clamp=clamp, sdsr=sdsr, tdsr=tdsr);
    (sum(fitvals)/denomsum, ml, fitvals, rerrs))
matchedfitval(X, GTX::AbstractVector, W, H; clamp=false, ordered=false) = (
            (ml, fitvals, rerrs, denomsum) = fitcomponents(X, GTX, W, H; clamp=clamp, ordered=ordered);
            (sum(fitvals)/denomsum, ml, fitvals, rerrs)
            )
matchednssd(GTW, GTH, W, H; clamp=false, sdsr=1, tdsr=1) = (
            (ml, mnssds, rerrs) = matchcomponents(GTW, GTH, W, H; clamp=clamp, dsr=dsr, tdsr=tdsr);
            (sum(mnssds)/length(mnssds), ml, mnssds, rerrs)
            )
function matchedimg(W, matchlist)
    Wmimg = zeros(size(W,1),length(matchlist))
    for mp in matchlist
        Wmimg[:,mp[1]] = W[:,mp[2]]
    end
    Wmimg
end

function ssdH(ml,gtH,H)
    ssd = 0.
    for (gti, i, invert) in ml
        if i>size(H,2) # no match found
            ssd += sum((gtH[:,gti]).^2)
        else
            ssd += invert ? sum((gtH[:,gti]+H[:,i]).^2) : sum((gtH[:,gti]-H[:,i]).^2)
        end
    end
    ssd
end

# match order with gtW, then W[:,nerorder] is same order with gtW
function matchedorder(ml,ncells)
    neworder = zeros(Int,length(ml))
    for (gti, i) in ml
        neworder[gti]=i
    end
    for i in 1:ncells
        i ∉ neworder && push!(neworder,i)
    end
    neworder
end
