load ../remote_mdls/debug1/Q-results.mat
load ../remote_mdls/debug3/Q-results.mat
load ../remote_mdls/qdebug_2state_3/Qnet.mat
size(Q)
size(voxel_grid)

figure(1)
for i=1:7
    subplot(3,3,i)
    p = Q(:,:,1,i);
    imagesc(p)
end

figure(2)

for i=1:7
    subplot(3,3,i)
    imagesc(Q(:,:,2,i))
end

figure(3)

for i=1:7
    subplot(3,3,i)
    imagesc(Q(:,:,3,i))
end

V = max(voxel_grid(:,9:14,:),[], 2);
V = reshape(V, [size(V,1),size(V,3)]);


%{
figure(3)
imagesc(V)
hold on

while 1
    x = round(ginput(1))
    Q(x(2), x(1), 2, :)
end
%}

figure(4)
imagesc(V)
