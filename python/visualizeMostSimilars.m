function visualizeMostSimilars(sup, exp_num, k)
    exp_num = num2str(exp_num)
    folder_name = ['/cs/vml2/mkhodaba/cvpr16/visualization/', num2str(exp_num),'-',num2str(sup),'-', num2str(k)]
    %load('/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/vw_commercial_vidinfo.mat')
    load('/cs/vml3/mkhodaba/cvpr16/dataset/vw_commercial/b1/03.mat')
    %load('/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/STM_similarities.mat')
    %load('/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/anna_color_similarities.mat')
    load(['/cs/vml2/mkhodaba/cvpr16/expriments/', exp_num ,'/similarities.mat'])
    %load('/cs/vml3/mkhodaba/cvpr16/Graph_construction/Features/allsegsvw_commercial.mat'])
    %similarities = -1 * similarities;
    mkdir(folder_name)
    row=similarities(sup,:);
    [vals, indices] = sort(row, 'descend');
    selected = zeros(1, length(indices));
    selected(indices(1:k)) = 1;
    maxx=max(row)
    minx=min(row)
    SpringColors=spring(5002);
    seg10 = labelledlevelvideo(:,:,1);
    img_seg10 = (cat(3,seg10,seg10,seg10));
    height = size(seg10,1)
    width = size(seg10, 2)
    
    for frame = 1:24
        frame
        current_frame_label=labelledlevelvideo(:,:,frame);
        %size(seg10)
        %type(labelledlevelvideo)
        for i = 1:height
            for j = 1:width
                sup_idx = current_frame_label(i,j);
                if sup_idx==sup
                    img_seg10(i,j,1)=0.1;
                    img_seg10(i,j,2)=0.1;
                    img_seg10(i,j,3)=1;
                else if selected(sup_idx) == 1 
                    value=row(sup_idx);
                    value=ceil(5000*(value-minx)/(maxx-minx))+1;
                    img_seg10(i,j,1)=SpringColors(value,1);
                    img_seg10(i,j,2)=SpringColors(value,2);
                    img_seg10(i,j,3)=SpringColors(value,3);
                else
                    img_seg10(i,j,1) = SpringColors(1,1);
                    img_seg10(i,j,2) = SpringColors(1,2);
                    img_seg10(i,j,3) = SpringColors(1,3);
                end
            end
        end
    imwrite(img_seg10,[folder_name, '/', num2str(frame), '.jpg']);
    end
end
%figure;
%histogram(row20);
