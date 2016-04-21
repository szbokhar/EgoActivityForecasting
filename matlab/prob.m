load ../remote_mdls/i500k_r100-100-30-0_b1_e07/Q-results.mat
load ../remote_mdls/i2M_r100-100-30-0_b1_e03/Q-results.mat
size(Q)
size(voxel_grid)

Q(umap==0) = min(Q(:))-1;
eQ = Q;
eQ = exp(eQ);

figure(1)
for i=1:9
    subplot(3,3,i)
    p = Q(:,:,1,i);
    imagesc(p)
end

figure(2)

for i=1:9
    subplot(3,3,i)
    imagesc(Q(:,:,2,i))
end
eQ = bsxfun(@rdivide, eQ, sum(eQ,4));

V = max(voxel_grid(:,6:13,:),[], 2);
V = reshape(V, [85,58]);
D = zeros([85, 58, 3])
tot = zeros([85, 58, 3])

figure(3)
colormap(gray)
imagesc(V)
hold on
lay = 1

allpos = []
pos = round(ginput(1))
D(pos(2), pos(1), 1) = 100

figure(4)
i = 0
prop = 0.5;
while 1
    i=i+1
    K = find(D);
    count = size(K,1)
    sum(D(:))

    [ys, xs, lst] = ind2sub(size(D), K);
    qk_left = sub2ind(size(eQ), ys, xs, lst, 2*ones([count, 1]));
    qk_down = sub2ind(size(eQ), ys, xs, lst, 3*ones([count, 1]));
    qk_right = sub2ind(size(eQ), ys, xs, lst, 4*ones([count, 1]));
    qk_up = sub2ind(size(eQ), ys, xs, lst, 5*ones([count, 1]));
    qk_hc = sub2ind(size(eQ), ys, xs, lst, 8*ones([count, 1]));

    K_up = sub2ind(size(D), ys-1, xs, lst);
    K_left = sub2ind(size(D), ys, xs-1, lst);
    K_down = sub2ind(size(D), ys+1, xs, lst);
    K_right = sub2ind(size(D), ys, xs+1, lst);
    K_hc = sub2ind(size(D), ys, xs, lst+1);


    nD = zeros(size(D));

    nD(K) = (1-prop)*D(K);
    nD(K_up) = nD(K_up) + prop*D(K).*eQ(qk_up);
    nD(K_left) = nD(K_left) + prop*D(K).*eQ(qk_left);
    nD(K_down) = nD(K_down) + prop*D(K).*eQ(qk_down);
    nD(K_right) = nD(K_right) + prop*D(K).*eQ(qk_right);
    nD(K_hc) = nD(K_hc) + prop*D(K).*eQ(qk_hc);

    size(find(D))

    D(D<0.001) = 0;
    nD(1,:,:) = 0;
    nD(end,:,:) = 0;
    nD(:,1,:) = 0;
    nD(:,end,:) = 0;
    nD(:,:,end) = 0;
    nD(isnan(nD)) = 0;
    nD = 100*nD ./ sum(nD(:));
    tot=tot+nD;


    




    subplot(1,2,1)
    imagesc(tot(:,:,1), [min(tot(:)), max(tot(:))])
    subplot(1,2,2)
    imagesc(tot(:,:,2), [min(tot(:)), max(tot(:))])

    D = nD;

    waitforbuttonpress
end

