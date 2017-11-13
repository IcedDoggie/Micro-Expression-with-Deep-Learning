%readFlowFile('1.flo')
image_counter = 1

path = "/home/vipr/Documents/CASME2_TIM/CASME2_TIM/"
flow_output = "/home/vipr/Documents/tvl1flow_3/flow/"
image_output = "/home/vipr/Documents/tvl1flow_3/optical_image/"

string_single_digit = "00"
string_double_digit = "0"

picture_ext = ".jpg"

image_array = []
array_for_flow_output = []
sub_counter = 1

while sub_counter <= 26
    if sub_counter >= 10
        sub_path = strcat("sub", int2str(sub_counter))
    else
        sub_path = strcat("sub", string_double_digit, int2str(sub_counter))
    end
    
    mkdir_str = [image_output, sub_path]
    mkdir(mkdir_str)
    
    subfolders = dir(strcat(path, sub_path))
    subfolders = subfolders([3:end])
    
    sub_counter ++
    
    for i=1:length(subfolders)
      video_path = subfolders(i).name
%      disp(video_path)
      mkdir_str = [image_output, sub_path, "/", video_path]
      mkdir(mkdir_str)
      image_counter = 1
      while image_counter < 11
        if image_counter >= 10
            string_to_parse = [path, sub_path, "/", video_path, "/", string_double_digit, int2str(image_counter), picture_ext]
            string_to_parse_for_flow = [flow_output, sub_path, "/", video_path, "/"]
            
        else
            string_to_parse = [path, sub_path, "/", video_path, "/", string_single_digit, int2str(image_counter), picture_ext]
            string_to_parse_for_flow = [flow_output, sub_path, "/", video_path, "/"]
        
        end
        
        image_array = strvcat(image_array, string_to_parse)
        array_for_flow_output = strvcat(array_for_flow_output, string_to_parse_for_flow)
        
        image_counter++
      
      end
    end
    
end



flow_counter = 1
ext = ".flo"
counter =   1
while counter <= length(array_for_flow_output)
  
  if mod(counter,10) != 0
  
    target_flow = [ array_for_flow_output(counter,:), int2str(counter), ext ] 
    target_flow = regexprep(target_flow, ' +', '')
    
    single_output = array_for_flow_output(counter, :)
    single_output = regexprep(single_output, 'flow/', 'optical_image/')
    
    output = readFlowFile(target_flow)
    of_x = output(:, :, 1)
    of_y = output(:, :, 2)
    write_x = [ single_output, int2str(flow_counter), "_x.png" ]
    write_y = [ single_output, int2str(flow_counter), "_y.png" ]
    
    
    % calculate flow magnitude and combine the whole thing into a 3-Channel Flow Image
    mag = sqrt(of_x.^2 + of_y.^2) % Euclidean Distance
    [x, y] = size(of_x)
    flow_image = zeros(x, y, 3)
    flow_image(:, :, 1) = of_x
    flow_image(:, :, 2) = of_y
    flow_image(:, :, 3) = mag
    write_flow = [ single_output, int2str(flow_counter), ".png" ]
    
    % remove white space
    write_x = strrep(write_x, '  ', '')
    write_y = strrep(write_y, '  ', '')
    write_flow = strrep(write_flow, '  ', '')
    
    
%    imwrite(of_x, write_x)
%    imwrite(of_y, write_y)
    imwrite(flow_image, write_flow)
  end
  
  flow_counter ++
  if flow_counter > 9
    flow_counter = 1
  end
  counter++
end
