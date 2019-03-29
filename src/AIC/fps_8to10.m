function data = fps_8to10(data)
frames = data(:,1);
data(:,1) = frames + floor(frames/4);
i = 1;
while i < length(data)
    frame = data(i,1);
    if mod(frame,4)==0
        frame = frame + 1;
        line = data(i,:);
        line(1) = frame;
        data = [data(1:i,:);line;data(i+1:end,:)];
    end
    i = i + 1;
end

end

