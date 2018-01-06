f=load('ILSVRC_val.mat');
filenames = f.filename;
class_labels = f.class_label;
for class_id = 0:999
    [r,c,~]=find(class_labels==class_id);
    if(isempty(r)== 0)
        dsp_str = sprintf('\n Processing class label %d with %d files',class_id, length(r));
        display(dsp_str)
        mkdir(strcat('../ILSVRC_data/Class_',num2str(class_id)));
        out_string ='../ILSVRC_data_ref/';
        for idx = 1:length(r)
            fl_str = sprintf('\n Filename : %s',filenames{r(idx)});
            display(fl_str)
            src_loc = strcat(out_string,filenames{r(idx)});
            dst_loc = strcat('../ILSVRC_data/Class_',num2str(class_id),'/',filenames{r(idx)});
            movefile(src_loc,dst_loc,'f');
 
        end

    end
end
