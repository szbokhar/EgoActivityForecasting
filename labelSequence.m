function acts = labelSequence(pathfile, imageroot, outfile)
    P = importdata(pathfile);
    strs = strcat(imageroot,P.textdata);

    acts = []
    for i=strs'
        imshow(i{1})
        waitforbuttonpress
        c = str2num(get(gcf, 'currentcharacter'))
        acts = [acts; c]

    end
    csvwrite(outfile, acts)
