function printmatrix(mat, label)

rows=size(mat,1);
outputstr = sprintf('%7.4f ',mat');

outputstr = reshape(outputstr,length(outputstr)/rows,rows)';

disp([label outputstr])
