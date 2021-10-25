This directory contains few of my implementations of Sinkhorn iterations for discrete optimal transport.

I have implemented a variant of sinhorn iterations for entropic regularized discrete optimal transport problem from [1] as given in [2]. 
Sinkhorn iterations are simultaneouly efficient and numerically unstable, which is the hallmark of interior point methods. 
I am interested in combining or entirely replacing trajectory based approach of interor-point with dynamically stable maps, which are usually numerically tractable. 
In that line, I found an interesting paper [3], which can be combined with mirror-descent based approach given in [4], for categorical distributions.






















[1] http://marcocuturi.net/Papers/cuturi13sinkhorn.pdf

[2] https://papers.nips.cc/paper/2017/file/491442df5f88c6aa018e86dac21d3606-Paper.pdf

[3] http://proceedings.mlr.press/v119/tong20a/tong20a.pdf

[4] http://proceedings.mlr.press/v108/guo20a/guo20a.pdf
