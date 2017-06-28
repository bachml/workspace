% prestep: use util_Generatefilelist.py to split standart file into $filename.txt 
% and $landmarks.txt 
% input arguments:
% preinput filelist as matrix
% preinput landmarks as matrix
% fileList = importdata('filelist.txt')
% points_input = load('landmarks.txt')
%%%%%%
base_points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041]';
%base_left_eye_x = 89.3095;
%base_left_eye_y = 72.9025;
%base_right_eye_x = 169.3095;
%base_right_eye_y = 72.9025;
%base_nose_x = 127.8949;
%base_nose_y = 127.0441;
%base_left_mouth_x  = 96.8796;
%base_left_mouth_y  = 184.8907;
%base_right_mouth_x  = 159.1065;
%base_right_mouth_y  = 184.7601;
%base_points = [base_left_eye_x base_left_eye_y; base_right_eye_x base_right_eye_y; base_nose_x base_nose_y;...
%   base_left_mouth_x base_left_mouth_y; base_right_mouth_x base_right_mouth_y];
%base_points = base_points / 2;

write_folder = 'cropped/';
mkdir(write_folder)
read_folder = '';
id_cropping = -1;
aligned_facial_mat = [];
id_flag = '';
for image_k = 1:6000
    if mod(image_k,200) == 0
        image_k
    end
    input_imagePath = [read_folder, fileList{image_k}];
    image = imread(input_imagePath);
    
    facial_temp = points_input(image_k, :);
    facial_X = facial_temp([1 3 5 7 9]);
    facial_Y = facial_temp([2 4 6 8 10]);
    facial_points = [facial_X; facial_Y]';
    
    tform = cp2tform(facial_points, base_points, 'nonreflective similarity');
    crop_image = imtransform(image, tform, 'XData', [1 96], 'YData', [1 112]);
    %crop_image = imtransform(image, tform, 'XData', [1 128], 'YData', [1 128]);
    output_imagePath = [write_folder, fileList{image_k}];
    name_index = strfind(output_imagePath, '/')
    save_imagePath = output_imagePath(1:name_index(length(name_index))-1)
    if ~exist(save_imagePath)
        mkdir(save_imagePath)
    end
    imwrite(crop_image,output_imagePath);
    
end
%U = facial_points(:,1);
%V = facial_points(:,2);
%[X, Y] = tformfwd(tform, U, V);
%X = round(X);
%Y = round(Y);
%aligned_facial_points = [X(1) Y(1) X(2) Y(2) X(3) Y(3) X(4) Y(4) X(5) Y(5)];
%aligned_facial_mat = [aligned_facial_mat; aligned_facial_points];