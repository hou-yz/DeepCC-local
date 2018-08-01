function result = intervalSearch( data, first, last )
%INTERVALSEARCH Returns indices of data elements that are within the range
%[first, last]
result = find((data >= first) & (data <= last));

end

