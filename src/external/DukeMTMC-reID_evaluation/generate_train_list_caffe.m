p = dir('./bounding_box_train/*.jpg');

txt = fopen('train_list.txt','w'); 
person_id = 0;
last_id = 1;
for i = 1:numel(p)
    file_path = sprintf('./bounding_box_train/%s',p(i).name);

    tmp_id = str2num(p(i).name(1:4));
    camera_id = str2num(p(i).name(7));
    
    if(tmp_id ~= last_id)
       person_id = person_id + 1;
       last_id = tmp_id;
    end
    fprintf(txt,'%s %d\n',file_path,person_id);  % caffe start from 0
end