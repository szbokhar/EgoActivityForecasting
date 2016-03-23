load ../remote_mdls/debug1/Q-results.mat
size(Q)
size(voxel_grid)

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

V = max(voxel_grid(:,8:15,:),[], 2);
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

figure(3)
imagesc(V)
hold on
lay = 1

allpos = []
pos = round(ginput(1))


while 1
    acts = Q(pos(2), pos(1), lay, :);
    acts = exp(acts);
    acts = acts/sum(acts(:))
    [v,i] = max(acts(:))

    if i == 1
        disp('nothing')
    elseif i == 2
        disp('left')
        pos(1) = pos(1)-1;
    elseif i == 3
        disp('down')
        pos(2) = pos(2)+1;
    elseif i == 4
        disp('right')
        pos(1) = pos(1)+1;
    elseif i == 5
        disp('up')
        pos(2) = pos(2)-1;
    elseif i == 6
        disp('ascend')
    elseif i == 7
        disp('descend')
    elseif i == 8
        disp('hc')
        lay = lay+1;
    elseif i == 9
        disp('end')
        break
    end
    allpos = [allpos; pos]
    l = size(allpos,1)
    if l >= 2
        plot(allpos((l-1):l,1), allpos((l-1):l,2), 'r-.')
    end

    waitforbuttonpress
end
