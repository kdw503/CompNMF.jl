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
fitd(a,b,na,nb) = (denom=na^2+nb^2+2na*nb; 1-sum(abs2,a-b)/denom)
calfit(a,b) = (dval=fitd(a,b); aval=fitd(a,-b); dval > aval ? (dval, false) : (aval, true))
calfit(a,b,na) = (dval=fitd(a,b,na); aval=fitd(a,-b,na); dval > aval ? (dval, false) : (aval, true))
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

function matchcomponents(GTW::AbstractArray{T}, GTH::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}; clamp=false) where T
    pq = PriorityQueue{Tuple{Int,Int,Bool}, T}(Base.Order.Forward) # Forward(low->high)
    gtcolnum = size(GTW,2); wcolnum = size(W,2)
    for i = 1:gtcolnum
        gtwi = GTW[:,i]; gthi = GTH[:,i]; gtxi = gtwi*gthi'
        for j = 1:wcolnum
            wj = W[:,j]; hj = H[j,:]; xj = wj*hj'
            clamp && (xj[xj.<0].=0)
            mnssd, invert = nssd(gtxi,xj)
            enqueue!(pq,(i,j,invert),mnssd)
        end
    end
    matchlist = Tuple{Int,Int,Bool}[]; ml = Int[]
    mnssds = T[]
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
            push!(mnssds,p[2])
        end
    end
    # Calculate unmatched power
    unmatchlist = collect(1:wcolnum)
    filter!(a->a ∉ ml,unmatchlist)
    gtxi = zeros(T,size(W,1),size(H,2))
    rerrs = T[]
    for j in unmatchlist
        wj = W[:,j]; hj = H[j,:]; xj = wj*hj'
        rerr = sum(abs2,xj)
        push!(rerrs,rerr)
    end
    matchlist, mnssds, rerrs
end

function fitcomponents(GTW::AbstractArray{T}, GTH::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T};
            clamp=false, iscalunmatched=false) where T
    pq = PriorityQueue{Tuple{Int,Int,Bool}, T}(Base.Order.Reverse) # Reverse(high->low)
    gtcolnum = size(GTW,2); wcolnum = size(W,2)
    for i = 1:gtcolnum
        gtwi = GTW[:,i]; gthi = GTH[:,i]; gtxi = gtwi*gthi'; ngtxi = norm(gtxi)
        for j = 1:wcolnum
            wj = W[:,j]; hj = H[j,:]; xj = wj*hj'
            clamp && (xj[xj.<0].=0)
            fitval = fitd(gtxi,xj,ngtxi)
            enqueue!(pq,(i,j,false),fitval)
        end
    end
    matchlist = Tuple{Int,Int,Bool}[]; ml = Int[]
    fitvals = T[]
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
            push!(fitvals,p[2])
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
    matchlist, fitvals, rerrs
end

matchedWnssd(GT,W) = ((ml, nssds) = matchcomponents(GT, W, nssd); (sum(nssds)/length(nssds), ml, nssds))
matchedWnssda(GT,W) = ((ml, nssdas) = matchcomponents(GT, W, nssda); (sum(nssdas)/length(nssdas), ml, nssdas))
matchedfitval(GTW, GTH, W, H; clamp=false, maskW=Colon(), maskH=Colon()) =
    ((ml, fitvals, rerrs) = fitcomponents(GTW, GTH, W, H; clamp=clamp);
    (sum(fitvals)/length(fitvals), ml, fitvals, rerrs))
matchednssd(GTW, GTH, W, H; clamp=false) = ((ml, mnssds, rerrs) = matchcomponents(GTW, GTH, W, H; clamp=clamp);
                    (sum(mnssds)/length(mnssds), ml, mnssds, rerrs))
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
