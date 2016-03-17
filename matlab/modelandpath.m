function modelandpath(pfile, seqfile, num_plot)

    P = importdata(pfile);
    rd = randperm(size(P,1),num_plot)';
    scatter3(P(rd,1), P(rd,2), P(rd,3), 1, P(rd,4:6)/255);
    hold on
    xlim([-60 60])
    ylim([-60 60])
    zlim([-60 60])

    S = importdata(seqfile)
    names = S.textdata
    S = S.data;

    for i=1:(size(S,1)-1)
        stp = i:(i+1);
        Q = S(stp,1:3);
        plot3(Q(:,1), Q(:,2), Q(:,3), 'r.-')

        waitforbuttonpress
    end
    hold off
end
