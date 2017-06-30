% prestep: use util_Generatefilelist.py to split standart file into $filename.txt 
% and $landmarks.txt 
% input arguments:
% preinput filelist as matrix
% preinput landmarks as matrix
% fileList = importdata('filelist.txt')
% points_input = load('landmarks.txt')
%%%%%%
%base_points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
%                51.6963, 51.5014, 71.7366, 92.3655, 92.2041]';
base_points = [89.3095, 169.3095, 127.8949, 96.8796, 159.1065; ...
                72.9025, 72.9025, 127.0441, 184.8907, 184.7601]';
num = size(points_input, 1);
aligned_landmarks = zeros(num, 10);
for image_k = 1:num
    if mod(image_k, 200) == 0
        image_k
    end
    facial_temp = points_input(image_k, :);
    facial_X = facial_temp([1 3 5 7 9]);
    facial_Y = facial_temp([2 4 6 8 10]);
    facial_points = [facial_X; facial_Y]';
    tform = cp2tform(facial_points, base_points, 'nonreflective similarity');
    
    U = facial_points(:,1);
    V = facial_points(:,2);
    [X, Y] = tformfwd(tform, U, V);
    X = round(X);
    Y = round(Y);
    aligned_facial_points = [X(1) Y(1) X(2) Y(2) X(3) Y(3) X(4) Y(4) X(5) Y(5)];
    %aligned_facial_mat = [aligned_facial_mat; aligned_facial_points];  
    aligned_landmarks(image_k, :) = aligned_facial_points;
end
save 238.mat aligned_landmarks
      