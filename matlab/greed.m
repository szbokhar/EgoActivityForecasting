load ../remote_mdls/i2M_r100-100-30-0_b1_e03/Q-results.mat
load ../remote_mdls/debug1/Q-results.mat
load ../remote_mdls/qdebug_2state_3/Qnet.mat

size(Q)
size(voxel_grid)

figure(1)
for i=1:size(Q,4)
    subplot(3,3,i)
    p = Q(:,:,1,i);
    imagesc(p)
end

figure(2)

for i=1:size(Q,4)
    subplot(3,3,i)
    imagesc(Q(:,:,2,i))
end

figure(3)

for i=1:size(Q,4)
    subplot(3,3,i)
    imagesc(Q(:,:,3,i))
end

size(Q)
V = sum(voxel_grid(:,5:10,:), 2);
V = reshape(V, [51,47]);


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
hold on
lay = 1

allpos = []
pos = round(ginput(1))

small = min(Q(:));
%Q(umap==0) = small-1;

while 1
    acts = Q(pos(2), pos(1), lay, :);
    acts = reshape(acts, [1,7])
    acts = exp(acts);
    acts = acts/sum(acts(:));
    [v,i] = max(acts(:))

    if i == 1
        disp('left')
        pos(1) = pos(1)-1;
    elseif i == 2
        disp('down')
        pos(2) = pos(2)+1;
    elseif i == 3
        disp('right')
        pos(1) = pos(1)+1;
    elseif i == 4
        disp('up')
        pos(2) = pos(2)-1;
    elseif i == 5
        disp('wash')
        lay = lay+1;
    elseif i == 6
        disp('hc')
        lay = lay+1;
    elseif i == 7
        disp('end')
        break
    end
    allpos = [allpos; pos];
    l = size(allpos,1)
    if l >= 2
        if lay == 1
            plot(allpos((l-1):l,1), allpos((l-1):l,2), 'r-.')
        elseif lay == 2
            plot(allpos((l-1):l,1), allpos((l-1):l,2), 'g-.')
        else
            plot(allpos((l-1):l,1), allpos((l-1):l,2), 'k-.')
        end
    end

    waitforbuttonpress
end
