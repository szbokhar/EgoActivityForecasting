load ../remote_mdls/dis_3statelog_i1M_r100_50_1_b1/Q-results.mat
load ../remote_mdls/debug1/Q-results.mat
load ../remote_mdls/qdebug_2state_3/Qnet.mat
size(Q)
size(voxel_grid)

eQ = Q;
eQ = exp(eQ);

figure(1)
for i=1:size(Q, 4)
    subplot(3,3,i)
    p = Q(:,:,1,i);
    imagesc(p)
end

figure(2)

for i=1:size(Q, 4)
    subplot(3,3,i)
    imagesc(Q(:,:,2,i))
end

figure(3)

for i=1:size(Q, 4)
    subplot(3,3,i)
    imagesc(Q(:,:,3,i))
end
eQ = bsxfun(@rdivide, eQ, sum(eQ,4));

V = sum(voxel_grid(:,5:10,:), 2);
V = reshape(V, [51,47]);
D = zeros([51, 47, 4])
tot = zeros([51, 47, 4])

colV = ind2rgb(gray2ind(V/max(V(:))), gray(255))

figure(4)
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
    qk_left = sub2ind(size(eQ), ys, xs, lst, 1*ones([count, 1]));
    qk_down = sub2ind(size(eQ), ys, xs, lst, 2*ones([count, 1]));
    qk_right = sub2ind(size(eQ), ys, xs, lst, 3*ones([count, 1]));
    qk_up = sub2ind(size(eQ), ys, xs, lst, 4*ones([count, 1]));
    qk_wsh = sub2ind(size(eQ), ys, xs, lst, 5*ones([count, 1]));
    qk_hc = sub2ind(size(eQ), ys, xs, lst, 6*ones([count, 1]));

    K_up = sub2ind(size(D), ys-1, xs, lst);
    K_left = sub2ind(size(D), ys, xs-1, lst);
    K_down = sub2ind(size(D), ys+1, xs, lst);
    K_right = sub2ind(size(D), ys, xs+1, lst);
    K_wsh = sub2ind(size(D), ys, xs, lst+1);
    K_hc = sub2ind(size(D), ys, xs, lst+1);


    nD = zeros(size(D));

    nD(K) = (1-prop)*D(K);
    nD(K_up) = nD(K_up) + prop*D(K).*eQ(qk_up);
    nD(K_left) = nD(K_left) + prop*D(K).*eQ(qk_left);
    nD(K_down) = nD(K_down) + prop*D(K).*eQ(qk_down);
    nD(K_right) = nD(K_right) + prop*D(K).*eQ(qk_right);
    nD(K_wsh) = nD(K_wsh) + prop*D(K).*eQ(qk_wsh);
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


    ctot1 = ind2rgb(gray2ind(tot(:,:,1)/max(tot(:)))*2, jet(255));
    ctot2 = ind2rgb(gray2ind(tot(:,:,2)/max(tot(:)))*2, jet(255));
    ctot3 = ind2rgb(gray2ind(tot(:,:,3)/max(tot(:)))*2, jet(255));




    subplot(1,3,1)
    imagesc(colV+ctot1)
    subplot(1,3,2)
    imagesc(colV+ctot2)
    subplot(1,3,3)
    imagesc(colV+ctot3)

    D = nD;

    waitforbuttonpress
end

