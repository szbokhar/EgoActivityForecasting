function modelandpath(pfile, seqfile, imgroot, num_plot)

    P = importdata(pfile);
    rd = randperm(size(P,1),num_plot)';
    subplot(2,1,1)
    scatter3(P(rd,1), P(rd,2), P(rd,3), 1, P(rd,4:6)/255);
    hold on
    xlim([-40 40])
    ylim([-40 40])
    zlim([-40 40])

    S = importdata(seqfile)
    names = S.textdata
    S = S.data;

    for i=1:(size(S,1)-1)
        subplot(2,1,1)
        disp(strcat(imgroot, names{i}))
        stp = i:(i+1);
        Q = S(stp,1:3);
        plot3(Q(:,1), Q(:,2), Q(:,3), 'r.-')

        subplot(2,1,2)
        imshow(imread(strcat(imgroot,names{i})))
        waitforbuttonpress
    end
    hold off
end
