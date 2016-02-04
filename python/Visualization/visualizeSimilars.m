function visualizeSimilars(sup, exp_num)
    exp_num = num2str(exp_num)
    folder_name = [num2str(sup), '-', exp_num]
    %load('/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/vw_commercial_vidinfo.mat')
    load('/cs/vml3/mkhodaba/cvpr16/dataset/vw_commercial/b1/03.mat')
    %load('/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/STM_similarities.mat')
    %load('/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/anna_color_similarities.mat')
    load(['/cs/vml2/mkhodaba/cvpr16/expriments/', exp_num ,'/similarities.mat'])
    %load('/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/allsegsvw_commercial.mat'])
    %similarities = -1 * similarities;
    mkdir(folder_name)
    row20=similarities(sup,:);
    size(row20)
    maxx=max(row20)
    minx=min(row20)
    SpringColors=spring(10002);
    for frame = 1:24
        seg10=labelledlevelvideo(:,:,frame);
        %size(seg10)
        %type(labelledlevelvideo)
        img_seg10 = (cat(3,seg10,seg10,seg10));
        for i = 1:size(seg10,1)
            for j = 1:size(seg10,2)
                if seg10(i,j)==sup
                    img_seg10(i,j,1)=0.1;
                    img_seg10(i,j,2)=0.1;
                    img_seg10(i,j,3)=1;
                else
                    idx = seg10(i,j);
                    value=row20(idx);
                    value=ceil(10000*(value-minx)/(maxx-minx))+1;
                    img_seg10(i,j,1)=SpringColors(value,1);
                    img_seg10(i,j,2)=SpringColors(value,2);
                    img_seg10(i,j,3)=SpringColors(value,3);
                end
            end
        end
    imwrite(img_seg10,[folder_name, '/', num2str(frame), '.jpg']);
    end
end
%figure;
%histogram(row20);
